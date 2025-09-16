#!/usr/bin/env python3
"""
Test script to verify query limit functionality
"""

import logging
import sys
from rag_system import TextToCypherRAG, QueryStep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_query_limits")

def test_query_limits():
    """Test that the system properly handles query limits"""
    
    try:
        # Initialize RAG system
        rag_system = TextToCypherRAG()
        
        # Mock schema
        mock_schema = {
            "vertices": [{"label": "TestNode", "attributes": []}],
            "edges": []
        }
        
        # Test with max_rounds = 3 (should force stop on 3rd round)
        max_rounds = 3
        question = "Test query with forced limits"
        previous_steps = []
        
        # Simulate first round
        logger.info(f"=== Testing Round 1 (should continue) ===")
        cypher, explanation, should_stop, prompt, llm_response = rag_system.generate_next_query(
            question, mock_schema, previous_steps, max_rounds
        )
        
        if should_stop:
            logger.error("❌ Round 1: System stopped unexpectedly")
            return False
        else:
            logger.info(f"✅ Round 1: Continuing as expected")
            logger.info(f"Generated query: {cypher}")
        
        # Add mock step for round 1
        step1 = QueryStep(1, "First query", cypher)
        step1.result = [{"test": "data1"}]
        previous_steps.append(step1)
        
        # Simulate second round 
        logger.info(f"=== Testing Round 2 (should continue with warning) ===")
        cypher, explanation, should_stop, prompt, llm_response = rag_system.generate_next_query(
            question, mock_schema, previous_steps, max_rounds
        )
        
        if should_stop:
            logger.error("❌ Round 2: System stopped unexpectedly")
            return False
        else:
            logger.info(f"✅ Round 2: Continuing as expected")
            logger.info(f"Generated query: {cypher}")
            # Check if warning about final round is in prompt
            logger.info(f"Round 2 prompt content: {prompt}")
            if "only have" in prompt and "round left" in prompt:
                logger.info("✅ Round 2: Warning about approaching limit found in prompt")
            else:
                logger.warning("⚠️ Round 2: No warning about approaching limit found")
        
        # Add mock step for round 2
        step2 = QueryStep(2, "Second query", cypher)
        step2.result = [{"test": "data2"}]
        previous_steps.append(step2)
        
        # Simulate third round (should force stop)
        logger.info(f"=== Testing Round 3 (should force stop) ===")
        cypher, explanation, should_stop, prompt, llm_response = rag_system.generate_next_query(
            question, mock_schema, previous_steps, max_rounds
        )
        
        logger.info(f"Round 3 prompt content: {prompt}")
        if "final round" in prompt and "must now STOP" in prompt:
            logger.info("✅ Round 3: Force stop message found in prompt")
        else:
            logger.warning("⚠️ Round 3: No force stop message found")
        
        if should_stop:
            logger.info(f"✅ Round 3: System stopped as expected")
            logger.info(f"Final answer: {explanation}")
            return True
        else:
            logger.error("❌ Round 3: System should have stopped but continued")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("🧪 Testing Query Limit functionality")
    logger.info("=" * 50)
    
    success = test_query_limits()
    
    logger.info("=" * 50)
    if success:
        logger.info("🎉 Query limit test PASSED!")
        sys.exit(0)
    else:
        logger.error("❌ Query limit test FAILED!")
        sys.exit(1)