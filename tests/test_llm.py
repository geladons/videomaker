import unittest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from modules.llm import generate_search_queries, plan_timeline, _clean_voiceover_text


class TestLLM(unittest.TestCase):
    def test_clean_voiceover_text_empty(self):
        result = _clean_voiceover_text("")
        self.assertEqual(result, "")

    def test_clean_voiceover_text_strips_narrator(self):
        result = _clean_voiceover_text("narrator: Hello world")
        self.assertEqual(result, "Hello world")

    def test_clean_voiceover_text_strips_quotes(self):
        result = _clean_voiceover_text('"Hello world"')
        self.assertEqual(result, "Hello world")

    def test_clean_voiceover_text_normalizes_whitespace(self):
        result = _clean_voiceover_text("Hello   world  test")
        self.assertEqual(result, "Hello world test")


class TestGenerateSearchQueries(unittest.TestCase):
    def test_fallback_on_empty_response(self):
        """Test that fallback queries are used when AI returns empty response."""
        scene = {"visual_query": "minecraft gameplay"}

        # Run async test
        async def run_test():
            with patch("httpx.AsyncClient.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"response": ""}
                mock_response.raise_for_status = MagicMock()
                mock_post.return_value = mock_response

                result = await generate_search_queries(
                    scene=scene,
                    asset_type="video",
                    model="test-model",
                    options={},
                    api_url="http://localhost:11434",
                    timeout=10.0,
                    request_delay=0.0,
                )
                # Should return fallback queries
                self.assertIsInstance(result, list)
                # Fallback should contain simplified query tokens
                self.assertTrue(len(result) > 0)

        asyncio.run(run_test())

    def test_fallback_on_exception(self):
        """Test that fallback queries are used when AI call raises exception."""
        scene = {"visual_query": "minecraft gameplay"}

        async def run_test():
            with patch("httpx.AsyncClient.post") as mock_post:
                mock_post.side_effect = Exception("Connection error")

                result = await generate_search_queries(
                    scene=scene,
                    asset_type="video",
                    model="test-model",
                    options={},
                    api_url="http://localhost:11434",
                    timeout=10.0,
                    request_delay=0.0,
                )
                # Should return fallback queries
                self.assertIsInstance(result, list)
                self.assertTrue(len(result) > 0)

        asyncio.run(run_test())


class TestPlanTimeline(unittest.TestCase):
    def test_fallback_timeline(self):
        """Test that fallback timeline is returned when LLM fails."""
        async def run_test():
            with patch("httpx.AsyncClient.post") as mock_post:
                mock_post.side_effect = Exception("Connection error")

                result = await plan_timeline(
                    prompt="Tell a story about minecraft",
                    model="test-model",
                    options={},
                    api_url="http://localhost:11434",
                    timeout=10.0,
                    target_duration=60.0,
                    request_delay=0.0,
                )
                # Should return fallback timeline
                self.assertIn("scenes", result)
                self.assertIn("title", result)
                self.assertEqual(len(result["scenes"]), 3)
                self.assertAlmostEqual(result["total_duration"], 60.0, delta=1.0)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()