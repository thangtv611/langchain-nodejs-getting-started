process.env.LANGCHAIN_CALLBACKS_BACKGROUND = true;
require('dotenv').config({path: '.env'});

const {ChatOllama} = require('@langchain/community/chat_models/ollama');
const {ChatPromptTemplate} = require('@langchain/core/prompts');
const {StringOutputParser} = require('@langchain/core/output_parsers');
const {
    CheerioWebBaseLoader,
} = require('@langchain/community/document_loaders/web/cheerio');
const {OllamaEmbeddings} = require('@langchain/community/embeddings/ollama');
const {RecursiveCharacterTextSplitter} = require('langchain/text_splitter');
const {MemoryVectorStore} = require('langchain/vectorstores/memory');
const {
    createStuffDocumentsChain,
} = require('langchain/chains/combine_documents');
const {createRetrievalChain} = require('langchain/chains/retrieval');
const {OpenAIEmbeddings, ChatOpenAI} = require('@langchain/openai');

const config = require('./config');

(async () => {
    try {
        const chatModel = new ChatOpenAI({
            ...config.openAI,
        });

        /* // ask single question with unknown role of the people who answers
        console.log(await chatModel.invoke('What is EventLoop term of Node.js?')); */

        /*
        // return the answer with context: which role of the answer is
        const outputParser = new StringOutputParser();
        const prompt = ChatPromptTemplate.fromMessages([
            [
                'system',
                'You are the world class technical documentation writter.',
            ],
            ['user', '{input}'],
        ]);

        const chain = prompt.pipe(chatModel).pipe(outputParser);
        console.log(
            await chain.invoke({ input: 'What is EventLoop term of Node.js?' })
        ); */

        const loader = new CheerioWebBaseLoader(
            'https://docs.smith.langchain.com/user_guide',
            // 'https://www.techinterviewhandbook.org/software-engineering-interview-guide/'
        );
        const docs = await loader.load();

        const splitter = new RecursiveCharacterTextSplitter();
        const splitDocs = await splitter.splitDocuments(docs);

        const prompt =
            ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context:
            <context>
            {context}
            </context>
    
            Question: {input}
        `);

        const embeddings = new OpenAIEmbeddings({
            ...config.openAI,
        });
        const vectorStore = await MemoryVectorStore.fromDocuments(
            splitDocs,
            embeddings
        );

        const retriever = vectorStore.asRetriever();

        const documentChain = await createStuffDocumentsChain({
            llm: chatModel,
            prompt,
        });

        const retrievalChain = await createRetrievalChain({
            combineDocsChain: documentChain,
            retriever,
        });

        const result = await retrievalChain.invoke({
            input: 'What is LangSmith used for? In term of AI',
        });

        console.log(result.answer);
    } catch (err) {
        console.error('ERROR: ', err);
        process.exit(0);
    }
})();
