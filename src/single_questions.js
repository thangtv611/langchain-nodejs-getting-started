process.env.LANGCHAIN_CALLBACKS_BACKGROUND = true;
require('dotenv').config({ path: '.env' });

const { ChatOllama } = require('@langchain/community/chat_models/ollama');
const { ChatPromptTemplate } = require('@langchain/core/prompts');
const { StringOutputParser } = require('@langchain/core/output_parsers');
const { CheerioWebBaseLoader } = require('@langchain/community/document_loaders/web/cheerio');
const { OllamaEmbeddings } = require('@langchain/community/embeddings/ollama');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const { MemoryVectorStore } = require('langchain/vectorstores/memory');
const { createStuffDocumentsChain } = require('langchain/chains/combine_documents');
const { createRetrievalChain } = require('langchain/chains/retrieval');
const { OpenAIEmbeddings, ChatOpenAI } = require('@langchain/openai');
const { MessagesPlaceholder } = require('@langchain/core/prompts');
const { createHistoryAwareRetriever } = require('langchain/chains/history_aware_retriever');
const { HumanMessage, AIMessage } = require('@langchain/core/messages');

const config = require('./config');
const utils = require('./utils');

utils.executor(async () => {
    const chatModel = new ChatOpenAI({
        ...config.openAI,
    });

    const loader = new CheerioWebBaseLoader('https://docs.smith.langchain.com/user_guide');
    const docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter();
    const splitDocs = await splitter.splitDocuments(docs);

    const prompt = ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context:
        <context>
        {context}
        </context>

        Question: {input}
    `);

    const embeddings = new OpenAIEmbeddings({
        ...config.openAI,
    });
    const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);

    const retriever = vectorStore.asRetriever();

    // answer with single question/answer
    const documentChain = await createStuffDocumentsChain({
        llm: chatModel,
        prompt,
    });

    const retrievalChain = await createRetrievalChain({
        combineDocsChain: documentChain,
        retriever,
    });

    const { answer } = await retrievalChain.invoke({
        input: 'What is LangSmith in term of AI?',
    });

    console.log(answer);
});
