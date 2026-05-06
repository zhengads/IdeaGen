"""
Literature Search Tool for InternAgent

This module provides tools for scientific literature search, citation management, and metadata extraction.
It integrates with multiple academic search engines and databases.
"""

import os
import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import aiohttp
import random
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass # 内置装饰器，自动生成 __init__ 和 self.xxx = xxx
class PaperMetadata: # 创建一个数据类 PaperMetadata，用于存储论文的元数据
    """Data class for paper metadata."""

    # 类的实例属性类型注解，定义了一个类中各个属性的名称、数据类型，以及默认值
    title: str
    authors: List[str]
    abstract: str
    year: Optional[int] = None
    doi: Optional[str] = None
    journal: Optional[str] = None
    url: Optional[str] = None
    citations: Optional[int] = None
    references: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    full_text: Optional[str] = None

    # 将一个类的实例属性转换为 Python 字典，方便数据的序列化、存储或传输
    # 创建一个字典，包含所有属性的属性名作为键、所有属性的值作为值，并返回该字典
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "year": self.year,
            "doi": self.doi,
            "journal": self.journal,
            "url": self.url,
            "citations": self.citations,
            "references": self.references,
            "keywords": self.keywords
        }

    #
    def to_citation(self, format_type: str = "apa") -> str:
        """
        Generate a formatted citation.
        
        Args:
            format_type: Citation format ("apa", "mla", "chicago", "harvard", "bibtex")
            
        Returns:
            Formatted citation string
        """
        if format_type == "apa": # APA format
            author_text = "" # 初始化作者文本为空字符串
            if self.authors: # 判断是否有作者属性并按 APA 规范处理作者（1 位 / 2 位 / 多位的不同写法）
                if len(self.authors) == 1:
                    author_text = f"{self.authors[0]}."
                elif len(self.authors) == 2:
                    author_text = f"{self.authors[0]} & {self.authors[1]}."
                else:
                    author_text = f"{self.authors[0]} et al."

            # 年份文本、期刊文本、DOI 文本
            year_text = f" ({self.year})." if self.year else ""
            journal_text = f" {self.journal}," if self.journal else ""
            doi_text = f" doi:{self.doi}" if self.doi else ""

            # 拼接所有部分，返回APA格式引用字符串
            return f"{author_text}{year_text} {self.title}.{journal_text}{doi_text}"
            
        elif format_type == "bibtex": # BibTeX format
            first_author = self.authors[0].split(" ")[-1] if self.authors else "Unknown" # 提取第一作者的姓氏（按空格分割取最后一段），无作者则为"Unknown"
            year = self.year or "Unknown" # 有年份则用年份，无则为"Unknown"
            key = f"{first_author}{year}" # 生成BibTeX格式的key
            
            authors = " and ".join(self.authors) if self.authors else "Unknown" # 作者列表用 "and" 连接（BibTeX规范），无作者则为"Unknown"

            # 拼接BibTeX格式字符串（多行拼接，保持格式规范）
            return (
                f"@article{{{key},\n"
                f"  author = {{{authors}}},\n"
                f"  title = {{{self.title}}},\n"
                f"  journal = {{{self.journal or 'Unknown'}}},\n"
                f"  year = {{{self.year or 'Unknown'}}},\n"
                f"  doi = {{{self.doi or ''}}}\n"
                f"}}" # 结尾与article后面的两个大括号形成闭合
            )
            
        # Default to a basic citation（非APA/BibTeX时，用基础格式）
        authors = ", ".join(self.authors) if self.authors else "Unknown" # 作者用逗号分隔
        year = f"({self.year})" if self.year else "" # 年份带括号，无则空
        journal = f"{self.journal}" if self.journal else "" # 期刊名，无则空
        
        return f"{authors} {year}. {self.title}. {journal}" # 返回基础格式



class CitationManager:
    """
    Manager for handling citations and bibliography.
    """
    
    def __init__(self):
        """Initialize the citation manager."""
        self.papers: Dict[str, PaperMetadata] = {} # 定义实例属性 papers，类型注解为「键是字符串、值是 PaperMetadata 实例」的字典，用于存储所有论文
        self.cached_search_results: Dict[str, List[PaperMetadata]] = {} # 定义实例属性 cached_search_results，类型注解为「键是字符串、值是 PaperMetadata 实例列表」的字典，用于缓存搜索结果
        
    def add_paper(self, paper: PaperMetadata) -> None: # 添加论文到管理器
        """
        Add a paper to the citation manager.
        
        Args:
            paper: Paper metadata to add
        """
        if paper.doi: # 有DOI则优先用DOI作为键
            self.papers[paper.doi] = paper
        else: # 无则用标题(Use title as key if no DOI)
            # Step 1：将标题转为小写并去除首尾空格，作为匹配关键词（避免大小写/空格导致的重复）
            key = paper.title.lower().strip()
            existing = False # 初始化标记为 False，默认没有重复
            
            # Step 2：遍历已存储的论文，检查是否有相同标题的论文（去重）
            for existing_paper in self.papers.values():
                if existing_paper.title.lower().strip() == key: # 已有的论文中存在相同标题的论文
                    existing = True # 标记为 True，表示有重复
                    break # 退出for循环，无需继续检查

            # Step 3：如果没有重复，则生成新键并存入（字典的添加就是先创建键再赋值，最终便形成key-value）
            if not existing:
                generated_key = f"paper_{len(self.papers)}" # 基于当前存储的论文数量生成键：paper_0、paper_1、paper_2...
                self.papers[generated_key] = paper # 添加新论文
    
    def clear(self) -> None: # 一次性重置管理器
        """Clear all papers from the manager."""
        self.papers.clear() # 清空存储的所有论文
        self.cached_search_results.clear() # 清空缓存的搜索结果



class LiteratureSearch:
    """
    Tool for searching scientific literature across multiple sources.
    """
    
    def __init__(self, 
                email: str, 
                api_keys: Optional[Dict[str, str]] = None,
                citation_manager: Optional[CitationManager] = None):
        """
        Initialize the literature search tool.
        
        Args:
            email: Email for API access (required for PubMed)
            api_keys: Dictionary of API keys for different sources
            citation_manager: Citation manager to use
        """
        self.email = email
        self.api_keys = api_keys or {}
        self.citation_manager = citation_manager or CitationManager()
        
        # Default search parameters
        self.default_max_results = 10
        self.default_sort = "relevance"  # or "date"
        
        # Cache for search results
        self._cache = {}




    async def search_arxiv(self, 
                         query: str, 
                         max_results: int = 10, 
                         sort: str = "relevance",
                         categories: Optional[List[str]] = None, # 可选参数，指定 arxiv 的论文分类（比如cs.AI），可以为None（搜索所有分类）
                         **kwargs) -> List[PaperMetadata]:
        """
        Search arXiv for papers matching the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            sort: Sort order ("relevance" or "date")
            categories: List of arXiv categories to search
            
        Returns:
            List of paper metadata
        """
        # Build the cache key
        cats_str = ",".join(categories) if categories else "all" # 如果传入了categories（非空列表），就用逗号拼接成字符串（比如["cs.AI", "physics.gen-ph"] → "cs.AI,physics.gen-ph"）；如果没传，就用"all"表示 “所有分类”
        cache_key = f"arxiv:{query}:{max_results}:{sort}:{cats_str}" # 拼接缓存键：用 f-string 把查询关键词、结果数量、排序方式、分类拼接成唯一的字符串（比如arxiv:machine learning:10:relevance:cs.AI），用于缓存查询结果。
        if cache_key in self._cache: # 如果缓存键存在，则使用缓存结果
            logger.info(f"Using cached results for arXiv query: {query}") # 输出日志信息，表示使用缓存结果
            return self._cache[cache_key] # 返回缓存结果
            
        logger.debug(f"Searching arXiv for: {query}") # 记录 “开始搜索” 的日志
        
        # arXiv API URL
        search_url = "http://export.arxiv.org/api/query"
        
        # Sort parameter
        sort_param = "relevance" if sort == "relevance" else "submittedDate" # 处理排序参数：arXiv API 要求的排序参数是relevance（相关性）或submittedDate（提交日期）
        
        # Category filter
        cat_filter = "" # 创建分类过滤器并初始为无过滤
        if categories: # 如果传入了分类参数
            cat_filter = " AND (" + " OR ".join([f"cat:{cat}" for cat in categories]) + ")" # 构造分类过滤器：用 "cat:" 前缀和 " OR " 连接多个分类（比如cs.AI OR physics.gen-ph）
            
        # Check if query already has field prefixes
        has_prefix = any(p in query for p in ["all:", "ti:", "abs:", "cat:", "au:"])
        
        if has_prefix:
            search_query_str = f"{query}{cat_filter}"
        else:
            # Avoid 'all:' to prevent matching author names like "Ai"
            # We wrap the whole query in parenthesis and apply to ti and abs
            search_query_str = f"(ti:{query} OR abs:{query}){cat_filter}"
        
        # Search parameters
        search_params = {
            "search_query": search_query_str,
            "max_results": max_results,
            "sortBy": sort_param, # 排序依据
            "sortOrder": "descending" # 降序排序
        }
        
        tries = 3 # 设置重试次数：最多尝试 3 次请求（防止网络波动导致单次请求失败）
        xml_data = None # Initialize xml_data to avoid UnboundLocalError
        for attempt in range(tries): # 循环尝试
            try: # try 块包裹可能出错的代码
                # Respect arXiv rate limits with a small jittered delay
                await asyncio.sleep(random.uniform(0.5, 1.5))
                
                async with aiohttp.ClientSession() as session: # 创建异步 HTTP 会话
                    async with session.get(search_url, params=search_params) as response: # 发送 GET 请求到 arXiv API，携带参数
                        if response.status != 200: # 检查响应状态码：非 200（成功）则记录错误日志，返回空列表
                            logger.error(f"arXiv search error: {response.status}")
                            if attempt < tries - 1: # 如果重试次数未达上限，则记录错误日志并等待 10 秒后重试
                                logger.info("Retrying in 10 seconds due to error...")
                                await asyncio.sleep(10)
                                continue # Correctly skip to the next iteration of the retry loop
                            else:
                                return [] # 返回空列表，结束函数
                        else:
                            xml_data = await response.text() # 异步读取响应的文本内容（arXiv API 返回 XML 格式的数据）
                            logger.info(f'arXiv REQUEST {query} success!') # 输出成功日志
                    
                        
                        if xml_data:
                            papers = self._parse_arxiv_xml(xml_data) # 将 XML 解析为 PaperMetadata 列表
                            
                            # Cache the results
                            self._cache[cache_key] = papers # 缓存结果：将解析后的论文列表存入缓存，下次相同搜索可直接使用
                            
                            logger.info(f"Get {len(papers)} papers from arXiv") # 输出日志信息，表示获取了多少篇论文
                            
                            # Add papers to citation manager
                            for paper in papers: # 遍历论文列表
                                self.citation_manager.add_paper(paper) # 添加论文到引用管理器
                                
                            return papers # 返回解析后的 PaperMetadata 列表
                        
            except Exception as e: # 捕获所有未预期的异常
                logger.error(f"Error searching arXiv: {e}") # 输出错误日志
                if attempt < tries - 1:
                    await asyncio.sleep(5)
                    continue
                return [] # 返回空列表，结束函数




        
    async def multi_source_search(self, 
                               query: str, 
                               sources: List[str] = None,
                               max_results: int = 10,
                               **kwargs) -> Dict[str, List[PaperMetadata]]:
        """
        Search multiple sources simultaneously.
        
        Args:
            query: Search query
            sources: List of sources to search
            max_results: Maximum results per source
            
        Returns:
            Dictionary mapping source names to result lists
        """
        if not sources:
            sources = ["arxiv"]

        # Prepare search tasks
        tasks = []
        for source in sources:
            if source == "arxiv":
                tasks.append(self.search_arxiv(query, max_results, **kwargs))

                
        # Execute all searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        combined_results = {}
        for source, result in zip(sources, results):
            if isinstance(result, Exception):
                logger.error(f"Error searching {source}: {str(result)}")
                combined_results[source] = []
            else:
                combined_results[source] = result
                
        return combined_results
    

    
    def _parse_arxiv_xml(self, xml_data: str) -> List[PaperMetadata]:
        """
        Parse arXiv XML response to extract paper metadata.
        
        Args:
            xml_data: XML response from arXiv
            
        Returns:
            List of paper metadata
        """
        papers = []
        soup = BeautifulSoup(xml_data, "xml")
        
        for entry in soup.find_all("entry"):
            try:
                # Title
                title_elem = entry.find("title")
                title_text = title_elem.text.strip() if title_elem else ""
                
                # Abstract
                summary_elem = entry.find("summary")
                abstract_text = summary_elem.text.strip() if summary_elem else ""
                
                # Authors
                authors = []
                for author in entry.find_all("author"):
                    name_elem = author.find("name")
                    if name_elem:
                        authors.append(name_elem.text.strip())
                
                # Publication year
                published_elem = entry.find("published")
                year = None
                if published_elem:
                    try:
                        pub_date = published_elem.text.strip()
                        match = re.search(r"(\d{4})", pub_date)
                        if match:
                            year = int(match.group(1))
                    except ValueError:
                        pass
                
                # DOI and URL
                doi = None
                url = None
                for link in entry.find_all("link"):
                    href = link.get("href", "")
                    if link.get("title") == "doi":
                        doi = href.replace("http://dx.doi.org/", "")
                    elif link.get("rel") == "alternate":
                        url = href
                
                # Create paper metadata
                paper = PaperMetadata(
                    title=title_text,
                    authors=authors,
                    abstract=abstract_text,
                    year=year,
                    doi=doi,
                    journal="arXiv",
                    url=url
                )
                papers.append(paper)
                
            except Exception as e:
                logger.error(f"Error parsing arXiv entry: {str(e)}")
        
        return papers
    
    def clear_cache(self) -> None:
        """Clear the search cache."""
        self._cache.clear()
