from __future__ import annotations

import unittest

def calculate_target_words(duration, wps):
    return max(3, int(duration * wps))

def calculate_min_words(target_words):
    return max(2, int(target_words * 0.7))

class TestTTSDensity(unittest.TestCase):
    def test_density_long_scene(self):
        # 10s scene at 2.0 WPS = 20 words
        duration = 10.0
        wps = 2.0
        target = calculate_target_words(duration, wps)
        self.assertEqual(target, 20)
        
        min_w = calculate_min_words(target)
        self.assertEqual(min_w, 14)

    def test_density_short_scene(self):
        # 2s scene at 2.0 WPS = 4 words
        duration = 2.0
        wps = 2.0
        target = calculate_target_words(duration, wps)
        self.assertEqual(target, 4)
        
        min_w = calculate_min_words(target)
        self.assertEqual(min_w, 2)

    def test_density_very_short_scene(self):
        # 1s scene at 2.0 WPS = 2 words, but floor is 3
        duration = 1.0
        wps = 2.0
        target = calculate_target_words(duration, wps)
        self.assertEqual(target, 3)
        
        min_w = calculate_min_words(target)
        self.assertEqual(min_w, 2)

if __name__ == "__main__":
    unittest.main()
