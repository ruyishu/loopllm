"""数学 ToolUse 评测（自动判分）：20 道必须借助 calculate_math 才算得准的题。
仓库自带的 eval_toolcall.py 只跑几个 demo、不打分；本脚本对齐 README 里
"full_sft 12/20 (60%) -> agent 17/20 (85%)" 的口径，给每个权重一个可比的分数。
工具协议、chat template、safe_math_eval 全部复用仓库实现，保证与训练时一致。"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import json
import time
import random
import argparse
import warnings
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params, safe_math_eval
from eval_toolcall import TOOLS as ALL_TOOLS, MOCK_RESULTS, parse_tool_calls, execute_tool, get_tools
warnings.filterwarnings('ignore')

# 工具集大小会决定 mini 模型会不会调工具：实测 8 个工具时 0 调用、1 个工具时正常。
# 默认只给 calculate_math（数学任务的最小充分集），另两种口径留给横向对照。
TOOL_MODES = {"math_only": ["calculate_math"],
              "math_plus": ["calculate_math", "get_current_time", "random_number"],
              "all": [t["function"]["name"] for t in ALL_TOOLS]}
TOOLS = get_tools(TOOL_MODES["math_only"])

# (题面, 参考表达式)。表达式只用于生成标准答案，不给模型看；题目固定，保证各权重之间可比。
MATH_TASKS = [
    ("帮我算一下 4837 乘以 926 等于多少？", "4837*926"),
    ("73921 加上 48675，再减去 12903，结果是多少？", "73921+48675-12903"),
    ("987654 减去 123456 等于多少？", "987654-123456"),
    ("8888 乘以 777 等于多少？", "8888*777"),
    ("1234567 加上 7654321 等于多少？", "1234567+7654321"),
    ("456789 的 3 倍是多少？", "456789*3"),
    ("999999 减去 888888 等于多少？", "999999-888888"),
    ("(256 加上 144) 乘以 33 等于多少？", "(256+144)*33"),
    ("1000000 除以 625 等于多少？", "1000000/625"),
    ("2 的 16 次方减去 100 等于多少？", "2**16-100"),
    ("15129 开平方，再加上 777，等于多少？", "sqrt(15129)+777"),
    ("8600 的 15% 是多少？", "8600*0.15"),
    ("(91 加 109) 乘以 (37 减 29) 等于多少？", "(91+109)*(37-29)"),
    ("7 的 9 次方等于多少？", "7**9"),
    ("98765 乘以 43 再加上 2109 等于多少？", "98765*43+2109"),
    ("123456789 除以 100 后向下取整是多少？", "floor(123456789/100)"),
    ("3.14159 乘以 2 再乘以 100 等于多少？", "3.14159*2*100"),
    ("一年（按 365 天算）一共有多少分钟？", "365*24*60"),
    ("我有 3 箱苹果，每箱 48 个，吃掉 25 个后还剩多少个？", "3*48-25"),
    ("一本书 365 页，每天读 13 页，读完需要多少天（向下取整）？", "floor(365/13)"),
]


def gold_of(expr):
    """标准答案：整数就返回 int，否则 float。"""
    v = safe_math_eval(expr)
    return int(v) if isinstance(v, int) or float(v).is_integer() else float(v)


def numbers_in(text):
    return [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', str(text).replace(',', '').replace('，', ''))]


def answer_match(text, gold):
    """整数答案要求精确命中；小数答案允许千分之一相对误差（模型可能自行四舍五入）。"""
    for v in numbers_in(text):
        if isinstance(gold, int):
            if abs(v - gold) < 0.5:
                return True
        elif abs(v - gold) <= max(1e-9, abs(gold) * 1e-3):
            return True
    return False


def init_model(args, weight):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    kwargs = {"hidden_size": args.hidden_size, "num_hidden_layers": args.num_hidden_layers,
              "use_moe": bool(args.use_moe)}
    if args.n_loops > 1:
        kwargs["n_loops"] = args.n_loops
    model = MiniMindForCausalLM(MiniMindConfig(**kwargs))
    moe_suffix = '_moe' if args.use_moe else ''
    ckp = f'./{args.save_dir}/{weight}_{args.hidden_size}{moe_suffix}.pth'
    model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer


def generate_text(model, tokenizer, messages, args):
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               tools=TOOLS, open_thinking=False)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(args.device)
    out = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"],
                         max_new_tokens=args.max_new_tokens, do_sample=True,
                         pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                         top_p=args.top_p, temperature=args.temperature)
    return tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)


def run_case(prompt, args, model, tokenizer):
    """跑满一轮多轮工具调用，返回 (所有 assistant 文本, 调用过 calculate_math?, 工具返回值, 轮数)。"""
    messages = [{"role": "user", "content": prompt}]
    assistant_texts, tool_results, calc_called, turns = [], [], False, 0
    for _ in range(args.max_turns):
        turns += 1
        content = generate_text(model, tokenizer, messages, args)
        assistant_texts.append(content)
        tool_calls = parse_tool_calls(content)
        if not tool_calls:
            break
        messages.append({"role": "assistant", "content": content})
        for tc in tool_calls:
            name = tc.get("name", "")
            if name == "calculate_math":
                calc_called = True
            result = execute_tool(tc)
            tool_results.append({"name": name, "call": tc, "result": result})
            if args.verbose:
                print(f'  📞 {name} | {json.dumps(tc.get("arguments", {}), ensure_ascii=False)}'
                      f' -> {json.dumps(result, ensure_ascii=False)}')
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
    return assistant_texts, calc_called, tool_results, turns


def eval_weight(model, tokenizer, args, weight):
    rows, correct = [], 0
    for i, (prompt, expr) in enumerate(MATH_TASKS[:args.n_questions]):
        gold = gold_of(expr)
        setup_seed(args.seed + i)  # 每题固定种子：不同权重面对同样的采样条件
        texts, calc_called, tool_results, turns = run_case(prompt, args, model, tokenizer)
        hit = answer_match('\n'.join(texts), gold)
        expr_ok = any(str(gold) in json.dumps(t.get("result", {}), ensure_ascii=False) for t in tool_results)
        correct += int(hit)
        rows.append({"i": i + 1, "prompt": prompt, "gold": gold, "hit": hit, "turns": turns,
                     "calc_called": calc_called, "expr_ok": expr_ok,
                     "answer": texts[-1].strip()[:200] if texts else ""})
        flag = '✅' if hit else '❌'
        print(f'{flag} [{i + 1:2d}/{len(MATH_TASKS[:args.n_questions])}] 标准={gold} 轮数={turns} '
              f'调用计算器={"是" if calc_called else "否"} 表达式正确={"是" if expr_ok else "否"}')
        if args.verbose:
            print(f'  💬 {prompt}\n  🧠 {texts[-1].strip()[:300] if texts else "(无输出)"}')
    n = len(rows)
    summary = {"weight": weight, "correct": correct, "n": n, "acc": round(100.0 * correct / max(n, 1), 1),
               "tool_mode": args.tool_mode,
               "calc_called": sum(r["calc_called"] for r in rows),
               "expr_ok": sum(r["expr_ok"] for r in rows),
               "avg_turns": round(sum(r["turns"] for r in rows) / max(n, 1), 2)}
    os.makedirs(f'./{args.save_dir}', exist_ok=True)
    detail_path = f'./{args.save_dir}/math_eval_{weight}_{args.tool_mode}.json'
    with open(detail_path, 'w', encoding='utf-8') as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f'============ 评测汇总 ============')
    print(f'weight={weight}  tool_mode={args.tool_mode}  correct={correct}/{n} ({summary["acc"]}%)  '
          f'calc_called={summary["calc_called"]}/{n}  expr_ok={summary["expr_ok"]}/{n}  '
          f'avg_turns={summary["avg_turns"]}')
    print(f'明细: {detail_path}')
    return summary


def main():
    parser = argparse.ArgumentParser(description="MiniMind 数学 ToolUse 评测（自动判分）")
    parser.add_argument('--load_from', default='../model', type=str, help="模型加载路径（model=原生torch权重）")
    parser.add_argument('--save_dir', default='../out', type=str, help="模型权重目录（明细也写这里）")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称，支持逗号分隔多个")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--n_loops', default=1, type=int, help="loop 层数（>1 时按 loop 模型加载）")
    parser.add_argument('--n_questions', default=len(MATH_TASKS), type=int, help="题目数量（最多 20）")
    parser.add_argument('--max_new_tokens', default=300, type=int, help="最大生成长度")
    parser.add_argument('--max_turns', default=4, type=int, help="最大多轮工具调用轮数")
    parser.add_argument('--tool_mode', default='math_only', choices=list(TOOL_MODES), type=str,
                        help="暴露给模型的工具集：math_only=仅计算器 / math_plus=3个 / all=仓库全部8个")
    parser.add_argument('--temperature', default=0.9, type=float, help="生成温度")
    parser.add_argument('--top_p', default=0.9, type=float, help="nucleus采样阈值")
    parser.add_argument('--seed', default=20260916, type=int, help="采样种子（每题 +题号）")
    parser.add_argument('--verbose', default=1, type=int, help="打印每题输出（0=只打分数）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()

    global TOOLS
    TOOLS = get_tools(TOOL_MODES[args.tool_mode])
    print(f'>>> 工具集 {args.tool_mode}: {[t["function"]["name"] for t in TOOLS]}')

    summaries = []
    for weight in [w.strip() for w in args.weight.split(',') if w.strip()]:
        print(f'>>> 权重 {weight}（n_loops={args.n_loops}）')
        model, tokenizer = init_model(args, weight)
        st = time.time()
        summaries.append(eval_weight(model, tokenizer, args, weight))
        print(f'>>> {weight} 用时 {time.time() - st:.0f}s')
        del model
        torch.cuda.empty_cache()
    print('============ 全权重汇总 ============')
    for s in summaries:
        print(f'{s["weight"]}: {s["correct"]}/{s["n"]} ({s["acc"]}%)')
    print('MATH_EVAL_DONE')


if __name__ == "__main__":
    main()
