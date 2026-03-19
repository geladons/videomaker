import unittest
from modules.compositor import _build_video_filters

class TestCompositor(unittest.TestCase):
    def test_build_video_filters_bg_video(self):
        # width, height, fps, bg_video, overlay_image, text_file, duration, font_name, font_size, input_offset
        filters, video_map = _build_video_filters(
            1920, 1080, 30, "bg.mp4", None, None, 10.0, "Arial", 24, 0
        )
        self.assertIn("setpts=PTS-STARTPTS", filters)
        self.assertNotIn("pts=STARTPTS", filters)
        print(f"Filters with bg_video: {filters}")

    def test_build_video_filters_no_bg_video(self):
        filters, video_map = _build_video_filters(
            1920, 1080, 30, None, None, None, 10.0, "Arial", 24, 0
        )
        self.assertIn("setpts=PTS-STARTPTS", filters)
        self.assertNotIn("pts=STARTPTS", filters)
        print(f"Filters without bg_video: {filters}")

if __name__ == "__main__":
    unittest.main()
