import asyncio
import unittest
from unittest.mock import AsyncMock, patch
from config import OLLAMA_API_URL

# Import after patching if possible, or patch where it's used
from orchestrator import Orchestrator

class TestSettingsInheritance(unittest.IsolatedAsyncioTestCase):
    @patch("orchestrator.get_setting", new_callable=AsyncMock)
    async def test_ai_query_inheritance(self, mock_get_setting):
        # Setup mocks
        async def side_effect(key, default=None):
            if key == "ollama_api_url":
                return "http://custom-ollama:11434"
            # Simulate others not being set
            return default

        mock_get_setting.side_effect = side_effect
        
        orch = Orchestrator(lambda tid, lvl, msg: asyncio.sleep(0))
        settings = await orch._get_effective_settings()
        
        # 'ai_query' key prefix is 'ollama_ai_query_'
        # It should fallback to OLLAMA_API_URL in the orchestrator loop
        self.assertEqual(settings["ai_query"]["api_url"], OLLAMA_API_URL)

    @patch("orchestrator.get_setting", new_callable=AsyncMock)
    async def test_ai_query_explicit(self, mock_get_setting):
        # Setup mocks
        async def side_effect(key, default=None):
            if key == "ollama_ai_query_api_url":
                return "http://ai-query-host:11434"
            return default

        mock_get_setting.side_effect = side_effect
        
        orch = Orchestrator(lambda tid, lvl, msg: asyncio.sleep(0))
        settings = await orch._get_effective_settings()
        
        self.assertEqual(settings["ai_query"]["api_url"], "http://ai-query-host:11434")

if __name__ == "__main__":
    unittest.main()
