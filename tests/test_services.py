import sys
import os
# Ensure the project root is in sys.path to allow importing backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import Mock, patch
# Import the function we are now testing
from backend.services import process_generation_request
# Import the class we need to mock
from backend.prompt_generation_pipline import PromptGenerationPipeline
# We might still need to mock the internal dependencies of the pipeline if the test structure required it,
# but with the current approach of mocking the pipeline itself, we don't need PromptGenerator or StyleExtractionAgent here.
# from backend.prompt_generator import PromptGenerator
# from backend.style_extraction_agent import StyleExtractionAgent

# Mock data for return values from pipeline methods
MOCK_GENERATED_PROMPTS = [
    "Prompt 1 generated by pipeline",
    "Prompt 2 generated by pipeline",
    "Prompt 3 generated by pipeline",
    "Prompt 4 generated by pipeline",
    "Prompt 5 generated by pipeline",
    "Prompt 6 generated by pipeline", # Assuming 6 prompts generated per round
]
MOCK_EXTRACTED_FEATURES_RETURN = {
    "0": {"style": ["minimalist"]},
    "1": {"subject": ["cat"]},
    "2": {"action": ["running"]},
    # ... features for other prompts ...
}
MOCK_PIPELINE_RETURN_VALUE = {
    "prompts": MOCK_GENERATED_PROMPTS,
    "extracted_features": MOCK_EXTRACTED_FEATURES_RETURN,
}

# Mock data for input feedback - NOW USES liked_prompts (list of strings)
MOCK_LIKED_PROMPT_STRINGS = [MOCK_GENERATED_PROMPTS[0], MOCK_GENERATED_PROMPTS[2]] # Simulate liking the 1st and 3rd prompts from the batch
MOCK_LIKED_STYLE_KEYWORDS = ["minimalist", "impressionist"]
MOCK_FEEDBACK = {
    "liked_prompts": MOCK_LIKED_PROMPT_STRINGS, # Updated key and value
    "liked_style_keywords": MOCK_LIKED_STYLE_KEYWORDS,
}

MOCK_USER_DESCRIPTION = "a dog jumping"


@pytest.fixture
def mock_pipeline():
    """Create a mock PromptGenerationPipeline instance."""
    pipeline = Mock(spec=PromptGenerationPipeline)

    # Mock the methods that process_generation_request will call
    pipeline.generate_initial_prompts.return_value = MOCK_PIPELINE_RETURN_VALUE
    pipeline.generate_refined_prompts.return_value = MOCK_PIPELINE_RETURN_VALUE
    pipeline.provide_feedback.return_value = None # provide_feedback doesn't return anything specific

    # Add a preferences attribute to control is_initial logic in the service function
    # This attribute needs to be a dictionary, mirroring the actual class
    pipeline.preferences = {"liked_prompts": []} # Initial state simulated

    return pipeline


# We are testing the process_generation_request function directly, which uses the pipeline object.

def test_process_generation_request_initial(mock_pipeline):
    """
    Test process_generation_request for initial generation (no feedback).
    """
    # Ensure the pipeline state indicates initial generation
    mock_pipeline.preferences = {"liked_prompts": []} # Simulate no previous liked prompts

    # Call the function
    result = process_generation_request(
        pipeline=mock_pipeline,
        user_description=MOCK_USER_DESCRIPTION,
        feedback=None # Simulate no feedback for initial generation
    )

    # Verify the expected results are returned (coming from the mock pipeline method)
    assert result == MOCK_PIPELINE_RETURN_VALUE

    # Verify provide_feedback was NOT called when feedback is None
    mock_pipeline.provide_feedback.assert_not_called()

    # Verify generate_initial_prompts was called with correct parameters
    mock_pipeline.generate_initial_prompts.assert_called_once_with(
        user_description=MOCK_USER_DESCRIPTION
    )

    # Verify generate_refined_prompts was NOT called
    mock_pipeline.generate_refined_prompts.assert_not_called()


def test_process_generation_request_refined_with_feedback(mock_pipeline):
    """
    Test process_generation_request for refined generation with feedback.
    """
    # Ensure the pipeline state indicates refined generation (e.g., some liked prompts exist)
    # This state change would typically happen AFTER a previous provide_feedback call,
    # but for this test simulating refined generation, we set the state directly on the mock.
    mock_pipeline.preferences = {"liked_prompts": ["some previous liked prompt string"]}


    # Call the function WITH feedback
    result = process_generation_request(
        pipeline=mock_pipeline,
        user_description=MOCK_USER_DESCRIPTION,
        feedback=MOCK_FEEDBACK # Simulate feedback with liked prompts (strings) and keywords
    )

    # Verify the expected results are returned (coming from the mock pipeline method)
    assert result == MOCK_PIPELINE_RETURN_VALUE

    # Verify provide_feedback was called with correct parameters - NOW EXPECTING STRINGS
    mock_pipeline.provide_feedback.assert_called_once_with(
        MOCK_FEEDBACK["liked_prompts"],       # Pass the liked prompt strings
        MOCK_FEEDBACK["liked_style_keywords"] # Pass the liked style keywords
    )

    # Verify generate_refined_prompts was called with correct parameters
    # The service function calls generate_refined_prompts, which uses the pipeline's
    # internal state (updated by provide_feedback) for context.
    mock_pipeline.generate_refined_prompts.assert_called_once_with(
        user_description=MOCK_USER_DESCRIPTION
    )

    # Verify generate_initial_prompts was NOT called
    mock_pipeline.generate_initial_prompts.assert_not_called()


def test_process_generation_request_refined_no_feedback(mock_pipeline):
    """
    Test process_generation_request for refined generation without feedback.
    (Should still call refined generation if pipeline state is not initial)
    """
    # Ensure the pipeline state indicates refined generation (e.g., some liked prompts exist)
    mock_pipeline.preferences = {"liked_prompts": ["some previous liked prompt string"]}


    # Call the function WITHOUT feedback
    result = process_generation_request(
        pipeline=mock_pipeline,
        user_description=MOCK_USER_DESCRIPTION,
        feedback=None # Simulate no feedback
    )

    # Verify the expected results are returned (coming from the mock pipeline method)
    assert result == MOCK_PIPELINE_RETURN_VALUE

    # Verify provide_feedback was NOT called
    mock_pipeline.provide_feedback.assert_not_called()

    # Verify generate_refined_prompts was called with correct parameters
    mock_pipeline.generate_refined_prompts.assert_called_once_with(
         user_description=MOCK_USER_DESCRIPTION
    )

    # Verify generate_initial_prompts was NOT called
    mock_pipeline.generate_initial_prompts.assert_not_called()


def test_process_generation_request_default_description(mock_pipeline):
    """
    Test process_generation_request with default user_description.
    """
    # Ensure initial state
    mock_pipeline.preferences = {"liked_prompts": []}

    # Call the function without user_description or feedback
    result = process_generation_request(
        pipeline=mock_pipeline,
        feedback=None # Simulate no feedback
    )

    # Verify the expected results are returned (coming from the mock pipeline method)
    assert result == MOCK_PIPELINE_RETURN_VALUE

    # Verify generate_initial_prompts was called with the default description
    mock_pipeline.generate_initial_prompts.assert_called_once_with(
        user_description="Generate a creative and visually appealing image" # Default value from service function
    )

    # Verify provide_feedback and generate_refined_prompts were NOT called
    mock_pipeline.provide_feedback.assert_not_called()
    mock_pipeline.generate_refined_prompts.assert_not_called()

# Note: We don't need app context mocks here because process_generation_request
# itself doesn't access current_app directly. It receives the pipeline object.
# The pipeline object is assumed to be correctly initialized elsewhere (e.g., within app context).