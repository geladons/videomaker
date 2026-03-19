from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

# Since get_setting is now the primary way to configure, we need a more robust mock
async def mock_get_setting(key, default=None):
    # This dictionary simulates the database settings
    mock_db = {
        "ollama_planner_model": "test-planner-model",
        "ollama_planner_params": {"temperature": 0.1},
        "ollama_planner_api_url": "http://planner-url",
        "ollama_planner_timeout": 60,
        "ollama_planner_think": False,
        "ollama_model": "test-default-model",
        "ollama_params": {"temperature": 0.5},
        "ollama_api_url": "http://default-url",
        "ollama_timeout": 120,
        "ollama_think": False,
        "ollama_request_delay": 0.0,
        "tts_engine": "piper",
        "voiceover_words_per_sec": 2.0,
    }
    return mock_db.get(key, default)

# Mock the database before importing Orchestrator to avoid connection issues
with patch('database.create_task', new=AsyncMock(return_value="task_123")):
    with patch('database.get_setting', side_effect=mock_get_setting):
        from orchestrator import Orchestrator

class TestDynamicTimeline(unittest.IsolatedAsyncioTestCase):
    
    async def _run_test_with_mocks(self, mock_plan_val, mock_get_dur_val, task_options):
        with (
            patch('orchestrator.llm.plan_timeline', new=AsyncMock(return_value=mock_plan_val)),
            patch('orchestrator.llm.generate_voiceover', new=AsyncMock(side_effect=lambda s, **kw: f"Voiceover for {s['voiceover']}")),
            patch('orchestrator.tts_engine.generate_voiceovers', new=AsyncMock(return_value=["/tmp/v1.wav", "/tmp/v2.wav"])),
            patch('orchestrator.get_wav_duration', side_effect=mock_get_dur_val),
            patch('orchestrator.os.path.exists', return_value=True),
            patch('orchestrator.scraper.gather_scene_assets', new=AsyncMock(return_value=[{"video": "v.mp4"} for _ in mock_plan_val["scenes"]])),
            patch('orchestrator.compositor.compose_video', new=AsyncMock(return_value="final.mp4")) as mock_compose,
            patch('orchestrator.subtitles.generate_ass', new=AsyncMock(return_value="subs.ass")),
            patch('orchestrator.scraper.download_cc_audio', new=AsyncMock(return_value="music.mp3")),
            patch('orchestrator.update_task_status', new=AsyncMock()),
            patch('orchestrator.add_log', new=AsyncMock()),
            patch('database.get_task', new=AsyncMock(return_value={"prompt": "Test", "options": task_options}))
        ):
            orc = Orchestrator(log_callback=AsyncMock())
            # We still need to mock these as they touch the real filesystem/DB
            orc._log = AsyncMock()
            orc._set_progress = AsyncMock()

            await orc._run_task("task_123")
            
            print(f"DEBUG LOGS: {orc._log.call_args_list}")
            return mock_compose.call_args

    async def test_dynamic_durations(self):
        """Audio duration should force scene expansion, ignoring target duration."""
        plan = {
            "title": "Test", "total_duration": 10.0, "music_mood": "happy",
            "scenes": [
                {"id": 1, "duration": 5.0, "voiceover": "Scene 1", "visual_query": "q1"},
                {"id": 2, "duration": 5.0, "voiceover": "Scene 2", "visual_query": "q2"}
            ]
        }
        # Scene 1 audio (8s) is longer than initial duration (5s)
        # Scene 2 audio (3s) is shorter
        audio_durs = [8.0, 3.0]
        task_opts = {"duration": 10.0}

        call_args = await self._run_test_with_mocks(plan, audio_durs, task_opts)
        
        self.assertIsNotNone(call_args, "mock_compose was NEVER called")
        final_scenes = call_args[0][0]

        # Scene 1 should be expanded to 8.3s (audio + padding)
        self.assertAlmostEqual(final_scenes[0]["duration"], 8.3, places=1)
        # Scene 2 should be expanded to 3.3s
        self.assertAlmostEqual(final_scenes[1]["duration"], 3.3, places=1)
        # Total duration is now ~11.6s, exceeding the target, which is correct
        self.assertGreater(sum(s["duration"] for s in final_scenes), 10.0)

    async def test_scale_up(self):
        """If total audio is shorter than target, scenes should scale up proportionally."""
        plan = {
            "title": "Test", "total_duration": 10.0, "music_mood": "happy",
            "scenes": [
                {"id": 1, "duration": 5.0, "voiceover": "S1", "visual_query": "q1"},
                {"id": 2, "duration": 5.0, "voiceover": "S2", "visual_query": "q2"}
            ]
        }
        # Audio is 1s each. After padding, scenes will be 1.3s each. Total = 2.6s.
        audio_durs = [1.0, 1.0]
        task_opts = {"duration": 10.0} # Target is 10s.

        call_args = await self._run_test_with_mocks(plan, audio_durs, task_opts)
        
        self.assertIsNotNone(call_args, "mock_compose was NEVER called")
        final_scenes = call_args[0][0]
        
        total = sum(s["duration"] for s in final_scenes)
        self.assertAlmostEqual(total, 10.0, places=1)
        # Each scene was 1.3s. Total 2.6. Scale factor is 10/2.6.
        # So each becomes 1.3 * (10 / 2.6) = 5.0
        self.assertAlmostEqual(final_scenes[0]["duration"], 5.0, places=1)
        self.assertAlmostEqual(final_scenes[1]["duration"], 5.0, places=1)

if __name__ == "__main__":
    unittest.main()
