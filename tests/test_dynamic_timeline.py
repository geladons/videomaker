from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

async def mock_get_setting(key, default=None):
    return default

# Mock the database before importing Orchestrator to avoid connection issues
with patch('database.create_task', new=AsyncMock(return_value="task_123")):
    with patch('database.get_setting', side_effect=mock_get_setting):
        from orchestrator import Orchestrator

class TestDynamicTimeline(unittest.IsolatedAsyncioTestCase):
    @patch('orchestrator.llm.plan_timeline')
    @patch('orchestrator.llm.generate_voiceover')
    @patch('orchestrator.tts_engine.generate_voiceovers')
    @patch('orchestrator.get_wav_duration')
    @patch('orchestrator.os.path.exists')
    @patch('orchestrator.scraper.gather_scene_assets')
    @patch('orchestrator.compositor.compose_video')
    @patch('orchestrator.subtitles.generate_ass')
    @patch('orchestrator.scraper.download_cc_audio')
    @patch('orchestrator.update_task_status')
    @patch('orchestrator.add_log')
    @patch('database.get_task')
    async def test_dynamic_durations(self, 
                                     mock_get_task,
                                     mock_add_log, 
                                     mock_update_status, 
                                     mock_download_cc,
                                     mock_gen_ass,
                                     mock_compose,
                                     mock_gather_assets,
                                     mock_exists, 
                                     mock_get_dur, 
                                     mock_gen_tts, 
                                     mock_gen_vo, 
                                     mock_plan):
        # Mock task from DB
        mock_get_task.return_value = {
            "prompt": "Test prompt",
            "options": {"duration": 10.0}
        }
        
        # Mock plan with 2 scenes, initial durations 5s each
        mock_plan.return_value = {
            "title": "Test Video",
            "total_duration": 10.0,
            "music_mood": "happy",
            "scenes": [
                {"id": 1, "duration": 5.0, "voiceover": "Scene 1", "visual_query": "query 1"},
                {"id": 2, "duration": 5.0, "voiceover": "Scene 2", "visual_query": "query 2"}
            ]
        }
        mock_gen_vo.side_effect = ["Voiceover 1 text", "Voiceover 2 text"]
        mock_gen_tts.return_value = ["/tmp/v1.wav", "/tmp/v2.wav"]
        mock_exists.return_value = True
        # Scene 1 audio takes 8s, Scene 2 audio takes 3s
        mock_get_dur.side_effect = [8.0, 3.0]
        mock_gather_assets.return_value = [{"video": "v1.mp4"}, {"video": "v2.mp4"}]
        mock_compose.return_value = "final.mp4"
        mock_gen_ass.return_value = "subs.ass"
        mock_download_cc.return_value = "music.mp3"

        orc = Orchestrator(log_callback=AsyncMock())
        # We need to mock the internal _log and _set_progress since they use database
        orc._log = AsyncMock()
        orc._set_progress = AsyncMock()

        await orc._run_task("task_123")

        # Capture the scenes that were passed to subtitles or compositor
        self.assertIsNotNone(mock_compose.call_args, "mock_compose was NEVER called")

        call_args = mock_compose.call_args[0]
        final_scenes = call_args[0]

        # Scene 1 should have been extended to 8.3s
        self.assertEqual(final_scenes[0]["duration"], 8.3)
        # Scene 2 should be exactly 3.3s because we forced audio-driven durations
        self.assertEqual(final_scenes[1]["duration"], 3.3)

    @patch('orchestrator.llm.generate_voiceover')
    @patch('orchestrator.tts_engine.generate_voiceovers')
    @patch('orchestrator.get_wav_duration')
    @patch('orchestrator.os.path.exists')
    @patch('orchestrator.scraper.gather_scene_assets')
    @patch('orchestrator.compositor.compose_video')
    @patch('orchestrator.subtitles.generate_ass')
    @patch('orchestrator.scraper.download_cc_audio')
    @patch('orchestrator.update_task_status')
    @patch('orchestrator.add_log')
    @patch('database.get_task')
    async def test_scale_up(self, 
                             mock_get_task,
                             mock_add_log, 
                             mock_update_status, 
                             mock_download_cc,
                             mock_gen_ass,
                             mock_compose,
                             mock_gather_assets,
                             mock_exists, 
                             mock_get_dur, 
                             mock_gen_tts, 
                             mock_gen_vo, 
                             mock_plan):
        # Mock task from DB
        mock_get_task.return_value = {
            "prompt": "Test prompt",
            "options": {"duration": 10.0}
        }
        
        mock_plan.return_value = {
            "title": "Short Audio Video",
            "total_duration": 10.0,
            "scenes": [
                {"id": 1, "duration": 5.0, "voiceover": "S1", "visual_query": "q1"},
                {"id": 2, "duration": 5.0, "voiceover": "S2", "visual_query": "q2"}
            ]
        }
        mock_gen_vo.side_effect = ["S1 text", "S2 text"]
        mock_gen_tts.return_value = ["v1.wav", "v2.wav"]
        mock_exists.return_value = True
        # Both audios take 1s -> min_dur = 1.3s
        mock_get_dur.side_effect = [1.0, 1.0]
        mock_gather_assets.return_value = [{"video": "v1.mp4"}, {"video": "v2.mp4"}]
        mock_compose.return_value = "final.mp4"
        mock_gen_ass.return_value = "subs.ass"
        mock_download_cc.return_value = "music.mp3"

        orc = Orchestrator(log_callback=AsyncMock())
        orc._log = AsyncMock()
        orc._set_progress = AsyncMock()
        
        await orc._run_task("task_456")

        call_args = mock_compose.call_args[0]
        final_scenes = call_args[0]
        
        # Total should hit 10.0s exactly
        total = sum(s["duration"] for s in final_scenes)
        self.assertAlmostEqual(total, 10.0, places=2)
        # Each should be 5.0s because (1.3 + 1.3) = 2.6. Target 10.0. Scale = 10/2.6.
        # 1.3 * (10/2.6) = 5.0
        self.assertEqual(final_scenes[0]["duration"], 5.0)
        self.assertEqual(final_scenes[1]["duration"], 5.0)

if __name__ == "__main__":
    unittest.main()
