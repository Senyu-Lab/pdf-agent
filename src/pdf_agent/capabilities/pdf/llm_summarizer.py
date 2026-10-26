
from pdf_agent.capabilities.pdf.chunker import TextChunk


def summarize_chunk_with_llm(llm, chunk: TextChunk) -> str:
    # 让模型只总结当前文本块，并保留重要信息
    prompt = (
        "请总结下面的 PDF 文本。"
        "保留重要事实、数据和结论，不要添加原文没有的信息。"
        f"\n原文页码：第 {chunk.start_page}-{chunk.end_page} 页"
        f"\n原文内容：\n{chunk.text}"
    )

    return llm.generate_text(prompt)


def combine_summaries_with_llm(llm, summaries_text: str) -> str:
    # 将分段摘要整合为连贯的完整文档摘要
    prompt = (
        "请将下面各段 PDF 摘要整合为一份连贯的完整文档摘要。"
        "保留主要事实、数据和结论，避免重复，不要添加没有依据的信息。"
        "请注意各段标注的原文页码。"
        f"\n分段摘要：\n{summaries_text}"
    )

    return llm.generate_text(prompt)
