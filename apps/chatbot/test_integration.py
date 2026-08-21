#!/usr/bin/env python3

import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(ENV_PATH)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("integration_test")

def test_imports():
    """Test that all required modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        import gradio as gr
        logger.info("✅ Gradio imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import Gradio: {e}")
        return False
    
    try:
        from backend import PuppyGraphChatbot
        logger.info("✅ Backend imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import backend: {e}")
        return False
    
    try:
        from rag_system import TextToCypherRAG
        logger.info("✅ RAG system imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import RAG system: {e}")
        return False
    
    return True

def test_rag_system():
    """Test RAG system functionality"""
    logger.info("Testing RAG system...")
    
    try:
        from rag_system import TextToCypherRAG, QueryExample
        
        # Initialize RAG system (will use lightweight model for testing)
        rag = TextToCypherRAG()
        logger.info("✅ RAG system initialized")
        
        # Test adding an example
        example = QueryExample(
            question="Integration test: count all nodes",
            cypher="MATCH (n) RETURN count(n) AS integration_node_count",
            description="Integration-test example"
        )
        count_before = rag.collection.count()
        if not rag.add_example(example):
            logger.error("❌ RAG system rejected the example")
            return False
        count_after = rag.collection.count()
        if count_after != count_before + 1:
            logger.error(
                "❌ Example was not stored: count stayed at %s", count_after
            )
            return False
        logger.info("✅ Example added to RAG system")
        
        # Test finding similar examples
        similar = rag.find_similar_examples("How many nodes are there?")
        if similar:
            logger.info(f"✅ Found {len(similar)} similar examples")
            return True
        else:
            logger.warning("⚠️ No similar examples found (may be expected)")
            return True
            
    except Exception as e:
        logger.error(f"❌ RAG system test failed: {e}")
        return False

def test_backend_basic():
    """Test basic backend functionality"""
    logger.info("Testing backend basic functionality...")
    
    try:
        from backend import PuppyGraphChatbot
        
        # The app requires a reachable PuppyGraph instance.
        try:
            chatbot = PuppyGraphChatbot()
            logger.info("✅ Backend initialized successfully")
            
            # Test schema retrieval
            schema = chatbot.get_schema()
            vertex_count = len(schema.get('vertices', []))
            edge_count = len(schema.get('edges', []))
            if vertex_count == 0 and edge_count == 0:
                logger.error("❌ PuppyGraph returned an empty or unsupported schema")
                return False
            logger.info(f"✅ Schema retrieved: {vertex_count} vertices, {edge_count} edges")
            
            # Test stats (this might fail if no connection to PuppyGraph)
            stats = chatbot.get_graph_stats()
            if "error" in stats:
                logger.error(f"❌ Graph stats returned error: {stats['error']}")
                return False
            else:
                logger.info(f"✅ Graph stats: {stats.get('node_count', 'unknown')} nodes, {stats.get('edge_count', 'unknown')} edges")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Backend connection failed: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Backend test failed: {e}")
        return False

def test_gradio_interface():
    """Test Gradio interface creation"""
    logger.info("Testing Gradio interface...")
    
    try:
        from gradio_app import create_interface
        
        # Create interface (don't launch)
        interface = create_interface()
        logger.info("✅ Gradio interface created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Gradio interface test failed: {e}")
        return False

def test_environment_setup():
    """Test environment and configuration"""
    logger.info("Testing environment setup...")
    
    # Check for .env file
    if ENV_PATH.exists():
        logger.info("✅ .env file found")
    else:
        logger.warning("⚠️ .env file not found (using .env.example)")
    
    # Check for Anthropic API key
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    if anthropic_key and not anthropic_key.startswith("your_"):
        logger.info("✅ Anthropic API key configured")
    else:
        logger.error("❌ Anthropic API key not configured")
        return False
    
    return True

def run_all_tests():
    """Run all integration tests"""
    logger.info("Starting PuppyGraph RAG Chatbot Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Module Imports", test_imports),
        ("RAG System", test_rag_system),
        ("Backend Basic", test_backend_basic),
        ("Gradio Interface", test_gradio_interface),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The system is ready to use.")
        return True
    else:
        logger.warning("⚠️ Some tests failed. Check the logs above for details.")
        return False

def main():
    """Main test runner"""
    success = run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
