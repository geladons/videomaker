import unittest
from unittest.mock import MagicMock, patch
from modules import compositor

class TestIntelligentRendering(unittest.TestCase):
    def test_build_ffmpeg_inputs_ss_offset(self):
        # Verify that -ss 15 is added before stock video input
        bg_video = "test_stock.mp4"
        inputs, audio_source, input_offset = compositor._build_ffmpeg_inputs(
            bg_video=bg_video,
            overlay_image=None,
            voiceover="voice.wav",
            resolution="1920x1080",
            fps=30,
            duration=5.0
        )
        
        self.assertIn("-ss", inputs)
        self.assertIn("15", inputs)
        # Ensure -ss 15 is BEFORE -i bg_video
        ss_idx = inputs.index("-ss")
        i_idx = inputs.index("-i")
        self.assertTrue(ss_idx < i_idx)
        self.assertEqual(inputs[i_idx + 1], bg_video)

    def test_build_transition_filters_math(self):
        # Verify xfade offsets for 3 scenes
        scene_files = ["s1.mp4", "s2.mp4", "s3.mp4"]
        durations = [5.0, 5.0, 5.0]
        trans_dur = 0.5
        
        filter_str, v_map, a_map = compositor._build_transition_filters(
            scene_files, durations, trans_dur
        )
        
        # Scene 0 to 1 transition offset = 5.0 - 0.5 = 4.5
        self.assertIn("offset=4.5", filter_str)
        # Scene 1 to 2 transition offset = (5.0 + 5.0) - 1.0 = 9.0
        self.assertIn("offset=9.0", filter_str)
        
        self.assertEqual(v_map, "[v2]")
        self.assertEqual(a_map, "[a2]")

    def test_build_transition_filters_short_scenes(self):
        # Verify that transition duration scales down for short scenes
        scene_files = ["s1.mp4", "s2.mp4"]
        durations = [0.4, 0.4] # Very short
        trans_dur = 0.5
        
        filter_str, v_map, a_map = compositor._build_transition_filters(
            scene_files, durations, trans_dur
        )
        
        # actual_trans_dur should be min(0.5, 0.4 / 2.1) ≈ 0.19
        self.assertIn("duration=0.19", filter_str)

    def test_build_transition_filters_single_scene(self):
        # Verify single scene case returns simple maps
        scene_files = ["s1.mp4"]
        durations = [5.0]
        
        filter_str, v_map, a_map = compositor._build_transition_filters(
            scene_files, durations
        )
        
        self.assertEqual(filter_str, "")
        self.assertEqual(v_map, "[0:v]")
        self.assertEqual(a_map, "[0:a]")

if __name__ == "__main__":
    unittest.main()
