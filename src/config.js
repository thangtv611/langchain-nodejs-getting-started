const path = require('path');
const envPath = path.resolve(process.cwd(), '.env');
require('dotenv').config({ path: envPath });

module.exports = {
    openAI: {
        apiKey: process.env.OPEN_AI_API_KEY,
    },
};
