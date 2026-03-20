import asyncio
import unittest
from unittest.mock import MagicMock, patch
from modules.llm import generate_voiceover, generate_search_queries, evaluate_asset_match
from modules.ai_helper import repair_json, summarize_error, call_llm_with_retry

class TestPromptLogging(unittest.IsolatedAsyncioTestCase):
    async def test_llm_logging(self):
        log_mock = MagicMock()
        async def log_side_effect(level, msg):
            log_mock(level, msg)
        
        # Mock httpx to avoid real API calls
        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"response": "test response"}
            mock_resp.status_code = 200
            
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post.return_value = mock_resp
            
            # Test generate_voiceover
            await generate_voiceover(
                scene={"voiceover": "test"},
                language="English",
                target_words=10,
                model="test",
                options={},
                api_url="http://test",
                timeout=10,
                log=log_side_effect
            )
            log_mock.assert_any_call("raw", unittest.mock.ANY)
            self.assertTrue(any("LLM Voiceover prompt:" in str(call) for call in log_mock.call_args_list))

            log_mock.reset_mock()
            # Test generate_search_queries
            await generate_search_queries(
                scene={"visual_query": "test"},
                asset_type="video",
                model="test",
                options={},
                api_url="http://test",
                timeout=10,
                log=log_side_effect
            )
            log_mock.assert_any_call("raw", unittest.mock.ANY)
            self.assertTrue(any("LLM Search queries prompt:" in str(call) for call in log_mock.call_args_list))

            log_mock.reset_mock()
            # Test evaluate_asset_match
            await evaluate_asset_match(
                visual_query="test",
                asset_description="test",
                model="test",
                options={},
                api_url="http://test",
                timeout=10,
                log=log_side_effect
            )
            log_mock.assert_any_call("raw", unittest.mock.ANY)
            self.assertTrue(any("Asset match prompt:" in str(call) for call in log_mock.call_args_list))

    async def test_ai_helper_logging(self):
        log_mock = MagicMock()
        async def log_side_effect(level, msg):
            log_mock(level, msg)
            
        with patch("httpx.AsyncClient") as mock_client:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"response": '{"test": "json"}'}
            mock_resp.status_code = 200
            
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post.return_value = mock_resp
            
            # Test repair_json
            await repair_json(
                raw_text="test",
                schema_hint="test",
                api_url="http://test",
                model="test",
                options={},
                timeout=10,
                log=log_side_effect
            )
            log_mock.assert_any_call("raw", unittest.mock.ANY)
            self.assertTrue(any("AI Repair prompt:" in str(call) for call in log_mock.call_args_list))

            log_mock.reset_mock()
            # Test summarize_error
            await summarize_error(
                raw_text="test",
                api_url="http://test",
                model="test",
                options={},
                timeout=10,
                log=log_side_effect
            )
            log_mock.assert_any_call("raw", unittest.mock.ANY)
            self.assertTrue(any("Error summary prompt:" in str(call) for call in log_mock.call_args_list))

            log_mock.reset_mock()
            # Test call_llm_with_retry
            await call_llm_with_retry(
                api_url="http://test",
                payload={"prompt": "test"},
                timeout=10,
                log=log_side_effect
            )
            log_mock.assert_any_call("raw", unittest.mock.ANY)
            self.assertTrue(any("LLM prompt (attempt 1):" in str(call) for call in log_mock.call_args_list))

if __name__ == "__main__":
    unittest.main()
