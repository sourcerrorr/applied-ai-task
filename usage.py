
_totals = {
    "input_tokens": 0,
    "output_tokens": 0,
    "query_n": 0
}

def record(usage):
    _totals['input_tokens'] += usage.prompt_tokens
    _totals['output_tokens'] += usage.completion_tokens
    _totals['query_n'] += 1


    
def get_totals():
    return dict(_totals)