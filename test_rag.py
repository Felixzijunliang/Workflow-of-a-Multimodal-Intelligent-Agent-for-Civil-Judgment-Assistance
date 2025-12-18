#!/usr/bin/env python3
"""
RAG系统快速测试脚本
用于验证RAG系统是否正常工作
"""
import requests
import time
import sys


def test_health():
    """测试健康检查接口"""
    print("=" * 50)
    print("测试1: 健康检查")
    print("=" * 50)
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ 服务状态: {data['status']}")
            print(f"✓ Qdrant连接: {data['qdrant_connected']}")
            print(f"✓ 模型加载: {data['model_loaded']}")
            print(f"✓ 集合名称: {data['collection_name']}")
            if data.get('vector_count'):
                print(f"✓ 向量数量: {data['vector_count']}")
            return True
        else:
            print(f"✗ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        print("提示: 请先运行 ./start_rag.sh 启动服务")
        return False


def test_stats():
    """测试统计接口"""
    print("\n" + "=" * 50)
    print("测试2: 统计信息")
    print("=" * 50)
    try:
        response = requests.get("http://localhost:8000/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ 集合名称: {data['collection_name']}")
            print(f"✓ 向量总数: {data['total_vectors']}")
            print(f"✓ 向量维度: {data['vector_dimension']}")
            print(f"✓ 距离度量: {data['distance_metric']}")
            return True
        else:
            print(f"✗ 获取统计失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return False


def test_search():
    """测试搜索接口"""
    print("\n" + "=" * 50)
    print("测试3: 语义搜索")
    print("=" * 50)

    test_query = "合同违约的赔偿责任"
    print(f"查询: {test_query}\n")

    try:
        response = requests.post(
            "http://localhost:8000/search",
            json={
                "query": test_query,
                "top_k": 3,
                "score_threshold": 0.0
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            print(f"✓ 找到 {data['count']} 个结果\n")

            if data['count'] > 0:
                for i, result in enumerate(data['results'], 1):
                    print(f"[{i}] 相似度: {result['score']:.4f}")
                    print(f"    来源: {result['source_file']}")
                    print(f"    内容: {result['text'][:100]}...")
                    print()
                return True
            else:
                print("⚠ 数据库中暂无数据")
                print("提示: 使用 vectorize_text.py 添加法律文本")
                return True
        else:
            print(f"✗ 搜索失败: {response.status_code}")
            print(f"   {response.text}")
            return False
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return False


def test_get_context():
    """测试获取RAG上下文接口"""
    print("\n" + "=" * 50)
    print("测试4: 获取RAG上下文")
    print("=" * 50)

    case_facts = """
    原告张三与被告李四于2023年1月签订房屋买卖合同，约定总价款100万元。
    被告仅支付首期款30万元，后两期款项均未按约定支付，构成违约。
    """

    print(f"案件事实: {case_facts.strip()}\n")

    try:
        response = requests.post(
            "http://localhost:8000/get_context",
            json={
                "case_facts": case_facts,
                "evidence_chain": "1. 房屋买卖合同 2. 银行转账记录",
                "top_k": 3,
                "min_score": 0.0
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            print(f"✓ 找到 {data['count']} 条相关法律\n")
            print("生成的上下文:")
            print("-" * 50)
            print(data['context'])
            print("-" * 50)
            return True
        else:
            print(f"✗ 获取上下文失败: {response.status_code}")
            print(f"   {response.text}")
            return False
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return False


def main():
    print("\n")
    print("╔" + "=" * 48 + "╗")
    print("║" + "    法律RAG系统测试脚本".center(48) + "║")
    print("╚" + "=" * 48 + "╝")
    print()

    results = []

    # 测试1: 健康检查
    results.append(("健康检查", test_health()))
    if not results[-1][1]:
        print("\n✗ 服务未启动，后续测试中止")
        sys.exit(1)

    time.sleep(1)

    # 测试2: 统计信息
    results.append(("统计信息", test_stats()))

    time.sleep(1)

    # 测试3: 搜索
    results.append(("语义搜索", test_search()))

    time.sleep(1)

    # 测试4: 获取上下文
    results.append(("RAG上下文", test_get_context()))

    # 总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")

    print()
    print(f"总计: {passed}/{total} 项测试通过")

    if passed == total:
        print("\n🎉 所有测试通过! RAG系统运行正常")
        sys.exit(0)
    else:
        print("\n⚠ 部分测试失败，请检查系统配置")
        sys.exit(1)


if __name__ == "__main__":
    main()
