from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from SPARQLWrapper import SPARQLWrapper2
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
import os
from typing import List
from rdflib import Literal
from rdflib.term import URIRef
from rdflib import Graph
from rdflib.namespace import NamespaceManager
from rdflib import Namespace
from pyvis.network import Network
load_dotenv(find_dotenv())


SUBGRAPH_RESPONSE = []


class GraphRAGRetriever(BaseRetriever):

    repo: str = "wine"
    url: str = "http://localhost:7200/repositories/"
    index: str = "my_index"
    predication_index = "triple_index"  # for triples similarity search. Optional here
    k: int
    t: float = 0.7

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        # Building SPARQL query for relevant documents
        endpoint = self.url + self.repo
        sparql_endpoint = SPARQLWrapper2(endpoint)

        # Il faut récupérer pour les 2 types de questions : exploratory question and connecting the dots

        relevant_query_entities = f"""
        PREFIX :<http://www.ontotext.com/graphdb/similarity/>
        PREFIX similarity-index:<http://www.ontotext.com/graphdb/similarity/instance/>
        SELECT ?documentID ?score {{
            ?search a similarity-index:{self.index} ;
            :searchTerm "{query}";
            :searchParameters "";
            :documentResult ?result .
            ?result :value ?documentID ;
            :score ?score .
        }} ORDER BY desc(?score)
        LIMIT {self.k}
        """

        most_relevant_entities = []

        try:
            print(
                f"The query send to the server for fecthing similar entities:\n\t {relevant_query_entities}\n\n")
            sparql_endpoint.setQuery(relevant_query_entities)
            result = sparql_endpoint.query().bindings
            if len(result) > 0:
                # We rerank the entities, some of them are not relevant
                for r in result:
                    if float(r["score"].value) < self.t:
                        continue
                    most_relevant_entities.append(r["documentID"].value)
                print(
                    f"Most relevants entities : \n\t{most_relevant_entities}\n\n")
            else:
                print("There is no RDF triples")

        except Exception as e:
            print(
                f"There is an error in the sparql request or in the endpoint URI: {e}")
            exit

        # Retrieve the bounded context for each entity
        documents = []

        for re in most_relevant_entities:
            sparql_bounded_context_query = f"""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                SELECT ?s ?p ?o
                WHERE {{
                    {{
                        ?s ?p <{re}> .
                    }}
                    UNION
                    {{
                        <{re}> ?p ?o .
                    }}
                }} LIMIT 20
            """

            """Important : we can also do semantic search for triples similar to a specfic triple or entity. See 'triple_indexing'"""

            try:
                print(
                    f"The query send to the server to get the bounded context of each entity:\n\t{sparql_bounded_context_query}\n\n")
                sparql_endpoint.setQuery(sparql_bounded_context_query)
                bounded_context_result = sparql_endpoint.query().bindings
                if len(bounded_context_result) > 0:
                    # we process the context of the entity

                    context_triples = []

                    for br in bounded_context_result:
                        subj = ""
                        pred = strip_prefix_from_uri(br["p"].value)
                        # None is return if the key doesn't exist
                        obj = br.get("o")

                        # if we have a subject and a predicat so the object will be the entity. Link going to the entity
                        if obj is None:
                            obj = strip_prefix_from_uri(re)
                            subj = strip_prefix_from_uri(br["s"].value)

                            context_triples.append((subj, pred, obj))
                            SUBGRAPH_RESPONSE.append(
                                (br["s"].value, br["p"].value, re))

                        # if we have an predicate and object so the subject is the entity. Link going out of the entity
                        else:
                            subj = strip_prefix_from_uri(re)
                            obj = strip_prefix_from_uri(br["o"].value)
                            context_triples.append((subj, pred, obj))
                            SUBGRAPH_RESPONSE.append(
                                (re, br["p"].value, br["o"].value))

                    # We construct fact
                    context_triple_string = "\n".join(
                        [f"{t[0], t[1], t[2]}" for t in context_triples])
                    context_triple_string = "{" + context_triple_string+"}"
                    documents.append(
                        Document(page_content=f"{re}: {context_triple_string}"))

            except Exception as e:
                print(
                    f"There is an error in the sparql request or in the endpoint URI: {repr(e)}")

        # we create the subgraph
        create_rdf_subgraph()
        return documents


def strip_prefix_from_uri(value):
    if "http" in value:
        uriOfValue = URIRef(value)
        graph = Graph()
        namespace_manager = graph.namespace_manager
        return namespace_manager.compute_qname(uriOfValue)[2]
    else:
        return str(value)


def create_rag_chain(user_question: str):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_GRAPH_RAG")
    llm = ChatOpenAI(api_key=OPENAI_API_KEY,
                     model="gpt-4o-mini", temperature=0)

    graph_retriever = GraphRAGRetriever(k=10)
    triples = graph_retriever._get_relevant_documents(query=user_question)

    system_message = """
    You are an assistant specialize in aswering user question based on RDF-like triples (facts). 
    The context below contains URIs and values in the form of key value-pair. 
    Each key corresponds to an URI and the value of that uri is a set of RDF-like triples, which correspond to the bounded context of an uri (node in the rdf graph).
    The format is : 
    <uri>: <{{(rdf-triples)}}>
    Using only the given Context below made of RDF-like triples, use it to answer the Question. Answer briefly.
    Dont include the folowing caracters : '<>()'
    *** Context ***
    {triples}
    """
    # Using only the given Context below, made of RDF-like triples, use it to answer the Question. Answer briefly

    human_message = """
    Answer to my question.
    ***Question***
    {question}
    """

    system_prompt = PromptTemplate.from_template(template=system_message)
    human_prompt = PromptTemplate.from_template(template=human_message)

    # Convert the list of documents to text
    messages = [SystemMessage(content=system_prompt.format(triples="\n".join([doc.page_content for doc in triples]))),
                HumanMessage(content=human_prompt.format(question=user_question))]

    return llm.invoke(messages).content


def create_rdf_subgraph():
    namespace_manager = NamespaceManager(Graph())
    wine = Namespace("http://www.ontotext.com/example/wine#")
    rdf = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns")
    rdfs = Namespace("http://www.w3.org/2000/01/rdf-schema#")
    xsd = Namespace("http://www.w3.org/2001/XMLSchema#")
    namespace_manager.bind("wine", wine)
    namespace_manager.bind("rdf", rdf)
    namespace_manager.bind("rdfs", rdfs)
    namespace_manager.bind("xsd", xsd)
    graph = Graph()
    graph.namespace_manager = namespace_manager

    for t in SUBGRAPH_RESPONSE:
        subj = URIRef(t[0])
        pred = URIRef(t[1])
        obj = t[2]
        if "http" not in obj:
            obj = Literal(obj)
        else:
            obj = URIRef(obj)

        graph.add((subj, pred, obj))

    graph.serialize(destination="./subgraph.ttl",
                    encoding="utf-8", format="ttl")

    net = Network(directed=True)

    for s, p, o in graph:
        subj_str = strip_prefix_from_uri(s)
        pred_str = strip_prefix_from_uri(p)
        obj_str = strip_prefix_from_uri(o)

        # Au moment où je génère le graph, éviter que ça aille sur le même noeud (le noeud se point vers lui-même)

        net.add_node(subj_str, label=subj_str, title=subj_str)
        net.add_node(obj_str, label=obj_str, title=obj_str)
        net.add_edge(subj_str, obj_str, title=pred_str)

    net.save_graph("./subgraph.html")
