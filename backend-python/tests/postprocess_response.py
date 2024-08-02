import re

def postprocess_response(s):
    REGEX_BLOCKS = r'```[\w]*(.*?)```'
    REGEX_FUNCTIONS = r'(\w+)*\('
    REGEX_ARGS = r'"([^"]+)"\s*=\s*"([^"]+)"'

    blocks = re.findall(REGEX_BLOCKS, s, re.DOTALL)
    print(f"Blocks:\n{blocks}")
    for block in blocks:
        functions = block.strip().split('\n')
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

if __name__ == '__main__':
    str = """
    some texts
    some texts
    some texts
    some texts

    ```python\n
    get_current_wether("location"= "Tokyo", "unit"= "None")\n
    ```

    some texts
    some texts
    some texts
    some texts
    """

    # str = """ get_exchange_rate
    # ```python
    # tool_call("base_currency"="feat(Backend)", "target_currency"="CNY"),
    # tool_call2("base_currency"="CNY", "target_currency"="USD"),
    # ```"""

    postprocess_response(str)
