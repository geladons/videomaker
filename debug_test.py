import asyncio
from unittest.mock import AsyncMock, patch
from tests.test_dynamic_timeline import TestDynamicTimeline, mock_get_setting

async def run_debug():
    test = TestDynamicTimeline()
    with patch('database.create_task', new=AsyncMock(return_value="task_123")), \
         patch('database.get_setting', side_effect=mock_get_setting):
        
        plan = {
            "title": "Test", "total_duration": 10.0, "music_mood": "happy",
            "scenes": [
                {"id": 1, "duration": 5.0, "voiceover": "Scene 1", "visual_query": "q1"},
                {"id": 2, "duration": 5.0, "voiceover": "Scene 2", "visual_query": "q2"}
            ]
        }
        audio_durs = [8.0, 3.0]
        task_opts = {"duration": 10.0}

        with patch('orchestrator.llm.plan_timeline', new=AsyncMock(return_value=plan)), \
             patch('orchestrator.llm.generate_voiceover', new=AsyncMock(side_effect=lambda s, **kw: f"Voiceover for {s['voiceover']}")), \
             patch('orchestrator.tts_engine.generate_voiceovers', new=AsyncMock(return_value=["/tmp/v1.wav", "/tmp/v2.wav"])), \
             patch('orchestrator.get_wav_duration', side_effect=audio_durs), \
             patch('orchestrator.os.path.exists', return_value=True), \
             patch('orchestrator.scraper.gather_scene_assets', new=AsyncMock(return_value=[{"video": "v.mp4"} for _ in plan["scenes"]])), \
             patch('orchestrator.compositor.compose_video', new=AsyncMock(return_value="final.mp4")) as mock_compose, \
             patch('orchestrator.subtitles.generate_ass', new=AsyncMock(return_value="subs.ass")), \
             patch('orchestrator.scraper.download_cc_audio', new=AsyncMock(return_value="music.mp3")), \
             patch('orchestrator.update_task_status', new=AsyncMock()), \
             patch('orchestrator.add_log', new=AsyncMock()), \
             patch('database.get_task', new=AsyncMock(return_value={"prompt": "Test", "options": task_opts})):

            from orchestrator import Orchestrator
            orc = Orchestrator(log_callback=AsyncMock())
            # Capture logs instead of mocking them completely
            log_msgs = []
            async def mock_log(task_id, level, msg):
                log_msgs.append(f"{level}: {msg}")
            orc._log = mock_log
            orc._set_progress = AsyncMock()

            await orc._run_task("task_123")
            
            for msg in log_msgs:
                print(msg)

asyncio.run(run_debug())
