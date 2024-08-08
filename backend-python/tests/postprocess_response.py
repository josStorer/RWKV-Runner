import re


def postprocess_response(s):
    REGEX_BLOCKS = r"([\w]+)[\s]*```[\w]*(.*?)```"
    REGEX_ARGS = r'"([^"]+)"\s*=\s*"([^"]+)"'

    name = re.search(REGEX_BLOCKS, s, re.DOTALL).group(1)
    function = re.search(REGEX_BLOCKS, s, re.DOTALL).group(2).strip()
    arguments = dict(re.findall(REGEX_ARGS, function))

    print(f"Name:\n{name}")
    print(f"Function:\n{function}")
    print(f"arguments:\n{arguments}")
    print()

    return


def postprocess_response_reserved(s):
    REGEX_BLOCKS = r"```[\w]*(.*?)```"
    REGEX_FUNCTIONS = r"(\w+)*\("
    REGEX_ARGS = r'"([^"]+)"\s*=\s*"([^"]+)"'

    blocks = re.findall(REGEX_BLOCKS, s, re.DOTALL)
    print(f"Blocks:\n{blocks}")
    for block in blocks:
        functions = block.strip().split("\n")
        print(f"Functions:\n{functions}")
        print()
        for function in functions:
            name = re.search(REGEX_FUNCTIONS, function).group(1)
            arguments = f"{dict(re.findall(REGEX_ARGS, function))}"

            print(function)
            print(name)
            print(arguments)
            print()

    return


if __name__ == "__main__":
    str = """
    some texts
    some texts
    some texts
    some texts

    ```python\n
    get_current_wether("location"= "Tokyo", "unit" ="None")\n
    ```

    some texts
    some texts
    some texts
    some texts
    """
    postprocess_response(str)

    str = """ get_exchange_rate
```python
tool_call("base_currency"= "func_as_param('Hello World!')", "target_currency"= "CNY")
```"""
    postprocess_response(str)

    str = """\
get_current_weather
```python\n
tool_call("location"= "Tokyo", "unit"= "None")\n
```"""
    postprocess_response(str)
