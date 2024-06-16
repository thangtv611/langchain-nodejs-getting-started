process.env.LANGCHAIN_CALLBACKS_BACKGROUND = true;
require('dotenv').config({ path: '.env' });

const { ChatPromptTemplate } = require('@langchain/core/prompts');
const { CheerioWebBaseLoader } = require('@langchain/community/document_loaders/web/cheerio');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const { MemoryVectorStore } = require('langchain/vectorstores/memory');
const { createStuffDocumentsChain } = require('langchain/chains/combine_documents');
const { createRetrievalChain } = require('langchain/chains/retrieval');
const { OpenAIEmbeddings, ChatOpenAI } = require('@langchain/openai');
const { MessagesPlaceholder } = require('@langchain/core/prompts');
const { createHistoryAwareRetriever } = require('langchain/chains/history_aware_retriever');
const { HumanMessage, AIMessage } = require('@langchain/core/messages');

const config = require('./config');

function executor(fn) {
    (async () => {
        try {
            fn();
        } catch (err) {
            console.error('[ERROR]: ', err);
        }
    })();
}

executor(async () => {
    const chatModel = new ChatOpenAI({
        ...config.openAI,
    });

    /** Load external document resources */
    const loader = new CheerioWebBaseLoader('https://docs.smith.langchain.com/user_guide');
    const docs = await loader.load();
    const splitter = new RecursiveCharacterTextSplitter();
    const splitDocs = await splitter.splitDocuments(docs);

    // Embedding document as vector data.
    const embeddings = new OpenAIEmbeddings({
        ...config.openAI,
    });

    // Vector data storage.
    const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
    const retriever = vectorStore.asRetriever();

    const historyAwarePrompt = ChatPromptTemplate.fromMessages([
        new MessagesPlaceholder('chat_history'),
        ['user', '{input}'],
        [
            'user',
            'Given the above conversation, generate a search query to look up in order to get information relevant to the conversation',
        ],
    ]);

    const historyAwareRetrieverChain = await createHistoryAwareRetriever({
        llm: chatModel,
        retriever,
        rephrasePrompt: historyAwarePrompt,
    });

    const historyAwareRetrievalPrompt = ChatPromptTemplate.fromMessages([
        ['system', "Answer the user's questions based on the below context:\n\n{context}"],
        new MessagesPlaceholder('chat_history'),
        ['user', '{input}'],
    ]);

    const historyAwareCombineDocsChain = await createStuffDocumentsChain({
        llm: chatModel,
        prompt: historyAwareRetrievalPrompt,
    });

    const conversationalRetrievalChain = await createRetrievalChain({
        retriever: historyAwareRetrieverChain,
        combineDocsChain: historyAwareCombineDocsChain,
    });

    const { answer } = await conversationalRetrievalChain.invoke({
        chat_history: [new HumanMessage('Can LangSmith help test my LLM applications?'), new AIMessage('Yes!')],
        input: 'tell me how by list it out',
    });

    console.log(answer);
});
