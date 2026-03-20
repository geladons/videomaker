import unittest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from modules.scraper import (
    _limit_query,
    _alternate_queries,
    generate_ai_search_queries,
    _build_scraper_constraints,
    _build_music_constraints,
)


class TestScraperHelpers(unittest.TestCase):
    def test_limit_query_short(self):
        result = _limit_query("minecraft gameplay", max_words=6)
        self.assertEqual(result, "minecraft gameplay")

    def test_limit_query_long(self):
        result = _limit_query("a screenshot of the original minecraft game", max_words=4)
        self.assertEqual(result, "a screenshot of the")

    def test_limit_query_empty(self):
        result = _limit_query("", max_words=6)
        self.assertEqual(result, "")


class TestAlternateQueries(unittest.TestCase):
    def test_basic_query(self):
        result = _alternate_queries("minecraft gameplay")
        self.assertIn("minecraft gameplay", result)
        self.assertTrue(len(result) > 0)

    def test_with_fallback_topic(self):
        result = _alternate_queries("screenshot", fallback_topic="minecraft")
        self.assertIn("minecraft", result)

    def test_empty_query(self):
        result = _alternate_queries("")
        self.assertTrue(len(result) > 0)  # Should have generic fallbacks

    def test_no_duplicates(self):
        result = _alternate_queries("minecraft gameplay")
        # Check no exact duplicates
        self.assertEqual(len(result), len(set(result)))

    def test_generic_fallbacks_included(self):
        result = _alternate_queries("")
        self.assertIn("nature landscape", result)
        self.assertIn("cinematic background", result)


class TestBuildConstraints(unittest.TestCase):
    def test_scraper_constraints(self):
        settings = {"yt_dlp_duration_filter": 300, "yt_dlp_download_section": 60}
        result = _build_scraper_constraints(settings)
        self.assertIn("--match-filter", result)
        self.assertIn("duration < 300", result)
        self.assertIn("--download-sections", result)
        self.assertIn("*0-60", result)

    def test_music_constraints_empty(self):
        result = _build_music_constraints()
        self.assertEqual(result, [])


class TestGenerateAISearchQueries(unittest.TestCase):
    def test_fallback_on_empty_response(self):
        """Test that fallback queries are used when AI returns empty response."""
        scene = {"visual_query": "minecraft gameplay"}

        async def run_test():
            with patch("httpx.AsyncClient.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"response": ""}
                mock_response.raise_for_status = MagicMock()
                mock_post.return_value = mock_response

                # Mock log function
                async def mock_log(level, msg):
                    pass

                result = await generate_ai_search_queries(
                    scene=scene,
                    asset_type="video",
                    log=mock_log,
                    api_url="http://localhost:11434",
                    model="test-model",
                    options={},
                    timeout=10.0,
                )
                # Should return fallback queries
                self.assertIsInstance(result, list)
                self.assertTrue(len(result) > 0)

        asyncio.run(run_test())

    def test_fallback_on_exception(self):
        """Test that fallback queries are used when AI call raises exception."""
        scene = {"visual_query": "minecraft gameplay"}

        async def run_test():
            with patch("httpx.AsyncClient.post") as mock_post:
                mock_post.side_effect = Exception("Connection error")

                async def mock_log(level, msg):
                    pass

                result = await generate_ai_search_queries(
                    scene=scene,
                    asset_type="video",
                    log=mock_log,
                    api_url="http://localhost:11434",
                    model="test-model",
                    options={},
                    timeout=10.0,
                )
                # Should return fallback queries
                self.assertIsInstance(result, list)
                self.assertTrue(len(result) > 0)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()