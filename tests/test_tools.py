from lambda_agent.tools import read_file


def test_read_file_success(tmp_path):
    # Create a temporary file
    test_file = tmp_path / "test.txt"
    test_content = "Hello, World!\nThis is a test file."
    test_file.write_text(test_content, encoding="utf-8")

    # Read the file using the tool
    result = read_file(str(test_file))

    # Verify the result
    assert result == test_content


def test_read_file_not_found():
    # Read a non-existent file
    result = read_file("non_existent_file_that_should_not_exist.txt")

    # Verify it returns an error string instead of throwing an exception
    assert result.startswith(
        "Error reading file non_existent_file_that_should_not_exist.txt: "
    )


def test_read_file_unicode(tmp_path):
    # Create a temporary file with unicode characters
    test_file = tmp_path / "unicode.txt"
    test_content = "Hello 🌍! Testing unicode: 안녕하세요."
    test_file.write_text(test_content, encoding="utf-8")

    # Read the file
    result = read_file(str(test_file))

    # Verify the unicode result
    assert result == test_content
