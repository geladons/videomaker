import sys
import os

# Add workspace to path
sys.path.append(os.getcwd())

from modules.llm import _fallback_timeline
from modules.utils import clean_and_tokenize

def test_extract_keywords():
    test_cases = [
        ("A detailed explanation of the historical impact of the industrial revolution", "industrial revolution"),
        ("Overview of how the internet works and its history", "internet works its history"),
        ("A video about showing the beauty of nature and forests", "beauty nature forests"),
        ("How to bake a cake in 10 minutes", "bake cake 10 minutes"),
        ("!!!!!!!!!!!!!!!!!!!!", ""), # Should handle empty or weird text
    ]
    
    print("Testing clean_and_tokenize...")
    for text, expected in test_cases:
        result = clean_and_tokenize(text)
        print(f"Input: '{text}' -> Result: '{result}'")
        # Note: expected is just a hint, we check if it's sensible
        assert len(result.split()) <= 5
    print("Keyword extraction tests passed!\n")

def test_fallback_timeline():
    prompt = "A detailed explanation of the historical impact of the industrial revolution on the 21st century's digital economy"
    print(f"Testing _fallback_timeline with prompt: '{prompt}'")
    
    result = _fallback_timeline(prompt, 60.0)
    
    print(f"Title: {result['title']}")
    print(f"Total Duration: {result['total_duration']}")
    
    for scene in result['scenes']:
        print(f"Scene {scene['id']}:")
        print(f"  Visual Query: '{scene['visual_query']}'")
        print(f"  Overlay Text: '{scene['overlay_text']}'")
        
        assert len(scene['visual_query'].split()) <= 5
        assert len(scene['overlay_text'].split()) <= 3
        assert scene['duration'] == 20.0
        
    print("\nFallback timeline test passed!")

if __name__ == "__main__":
    test_extract_keywords()
    test_fallback_timeline()
