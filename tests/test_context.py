from lambda_agent.context import trim_chat_history, clip


class MockFunctionResponseData:
    def __init__(self, result_value):
        self.response = {"result": result_value}


class MockFunctionResponse:
    def __init__(self, response_data):
        self.function_response = response_data


class MockPart:
    def __init__(self, function_response=None):
        self.function_response = function_response


class MockContent:
    def __init__(self, parts):
        self.parts = parts


def create_history_with_functions(num_functions, text_length=1000):
    history = []
    for i in range(num_functions):
        # Interleave with some non-function content
        history.append(MockContent(parts=[MockPart()]))

        # Add function response content
        fr_data = MockFunctionResponseData("a" * text_length)
        part = MockPart(function_response=fr_data)
        history.append(MockContent(parts=[part]))
    return history


def test_clip():
    assert clip("hello", 10) == "hello"
    assert clip("hello world", 5) == "hello\n...[TRUNCATED — original 11 chars]"


def test_trim_chat_history_tiers():
    # Create 15 function responses, each with 1000 characters
    history = create_history_with_functions(15, 1000)

    # Apply default trims explicitly to test the unbounded Tier 1 behavior
    # TIER1_COUNT = 4, TIER1_LIMIT = None
    # TIER2_COUNT = 8, TIER2_LIMIT = 180
    # TIER3_LIMIT = 80
    trim_chat_history(history, tier1_limit=None)

    # Extract results
    results = []
    for content in history:
        for part in content.parts:
            if getattr(part, "function_response", None):
                results.append(part.function_response.response["result"])

    # results are ordered oldest to newest.
    # The last 4 (most recent) should be TIER1_LIMIT (None -> 1000 chars)
    # The previous 8 should be TIER2_LIMIT (180 + truncation message)
    # The first 3 should be TIER3_LIMIT (80 + truncation message)

    assert len(results) == 15

    # Check most recent 4 (Tier 1)
    for i in range(11, 15):
        assert len(results[i]) == 1000
        assert "TRUNCATED" not in results[i]

    # Check next 8 (Tier 2)
    for i in range(3, 11):
        assert results[i].startswith("a" * 180)
        assert "TRUNCATED — original 1000 chars" in results[i]

    # Check oldest 3 (Tier 3)
    for i in range(0, 3):
        assert results[i].startswith("a" * 80)
        assert "TRUNCATED — original 1000 chars" in results[i]


def test_trim_chat_history_custom_limits():
    history = create_history_with_functions(3, 500)

    # Custom limit: tier1_count=1, tier1_limit=10, tier2_count=1, tier2_limit=5, tier3_limit=2
    trim_chat_history(
        history,
        tier1_count=1,
        tier1_limit=10,
        tier2_count=1,
        tier2_limit=5,
        tier3_limit=2,
    )

    results = []
    for content in history:
        for part in content.parts:
            if getattr(part, "function_response", None):
                results.append(part.function_response.response["result"])

    assert len(results) == 3
    # Oldest (Tier 3)
    assert results[0].startswith("a" * 2)
    # Middle (Tier 2)
    assert results[1].startswith("a" * 5)
    # Newest (Tier 1)
    assert results[2].startswith("a" * 10)


def test_trim_chat_history_no_function_responses():
    history = [MockContent(parts=[MockPart()]), MockContent(parts=[MockPart()])]
    trim_chat_history(history)
    # Should not crash and do nothing
    assert len(history) == 2
