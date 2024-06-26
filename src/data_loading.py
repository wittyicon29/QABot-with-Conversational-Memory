from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.document_loaders.merge import MergedDataLoader

def load_data():
    web_urls = [
        "https://stanford-cs324.github.io/winter2022/lectures/introduction/",
        "https://stanford-cs324.github.io/winter2022/lectures/capabilities/",
        "https://stanford-cs324.github.io/winter2022/lectures/harm-1/",
        "https://stanford-cs324.github.io/winter2022/lectures/harm-2/",
        "https://stanford-cs324.github.io/winter2022/lectures/data/",
        "https://stanford-cs324.github.io/winter2022/lectures/modeling/",
        "https://stanford-cs324.github.io/winter2022/lectures/training/"
    ]
    web_loaders = [WebBaseLoader(url) for url in web_urls]
    loader_pdf = PyPDFLoader("Milestone papers.pdf")
    loader_all = MergedDataLoader(loaders=web_loaders + [loader_pdf])
    return loader_all.load()
