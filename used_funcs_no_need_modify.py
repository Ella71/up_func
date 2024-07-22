def generate_combinations_recursive(lst, current_comb=[], start=0, all_combinations=None):
    if all_combinations is None:
        all_combinations = []

    if start == len(lst):
        if current_comb:  # 避免添加空组合
            all_combinations.append(current_comb)
        return all_combinations
    # 不选当前元素，递归调用
    generate_combinations_recursive(lst, current_comb, start + 1, all_combinations)
    # 选当前元素，递归调用
    new_comb = current_comb + [lst[start]]
    generate_combinations_recursive(lst, new_comb, start + 1, all_combinations)
    return all_combinations
