const { OpenAIEmbeddings } = require('@langchain/openai');
const config = require('./config');

const embeddings = new OpenAIEmbeddings({ ...config.openAI });

(async () => {
    const res = await embeddings.embedQuery('Hello World');
    console.log('🚀 ~ res:', res);

    const documentRes = await embeddings.embedDocuments(["Hello world", "Bye bye"]);
    console.log("🚀 ~ documentRes:", documentRes)
})();
