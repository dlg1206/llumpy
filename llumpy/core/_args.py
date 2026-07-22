"""
File: _args.py

Description: Util methods to support prompting LLMs

@author Derek Garcia
"""


def _read_file(file: str) -> str:
    """
    Read content of a file

    :param file: Path to file
    :return: File content
    """
    with open(file, 'r', encoding='utf-8') as f:
        return f.read()


def _validate_args(content: str | None, file: str | None) -> str:
    """
    Validate LLM prompt args

    :param content: LLM content arg
    :param file: File path to LLM content
    :raises ValueError: If nether or both content or file provided
    :return: File content or content
    """
    # neither provided
    if content is None and file is None:
        raise ValueError("Either content or file must be provided")
    # both provided
    if content is not None and file is not None:
        raise ValueError("Only one of content or file may be provided")
    # return content if set
    if content is not None:
        return content
    # return file contents if set
    if file is not None:
        return _read_file(file)
    # should not reach here
    raise RuntimeError("unreachable: neither content nor file is set")
