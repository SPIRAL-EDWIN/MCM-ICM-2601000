const https = require('https');

const API_KEY = "sk_MWhPzZXMBVJeKcnAO9hL-cxQDmM7t2aMknf8tC_6tbQ";
const BASE_URL = "https://api.jiekou.ai/openai/v1/chat/completions";

function testModel(model) {
    return new Promise((resolve) => {
        console.log(`\n--- Testing Model: ${model} ---`);
        
        const data = JSON.stringify({
            model: model,
            messages: [{role: "user", content: "Say hello."}],
            max_completion_tokens: 100
        });

        const url = new URL(BASE_URL);
        const options = {
            hostname: url.hostname,
            path: url.pathname,
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${API_KEY}`,
                'Content-Type': 'application/json',
                'Content-Length': data.length
            }
        };

        const req = https.request(options, (res) => {
            let body = '';
            res.on('data', (chunk) => body += chunk);
            res.on('end', () => {
                if (res.statusCode === 200) {
                    try {
                        const parsed = JSON.parse(body);
                        console.log("✅ Success!");
                        console.log(`Response Body: ${body}`);
                        console.log(`Response: ${parsed.choices[0].message.content}`);
                        resolve(true);
                    } catch (e) {
                        console.log(`❌ Parse Error: ${e.message}`);
                        resolve(false);
                    }
                } else {
                    console.log(`❌ Failed (Status ${res.statusCode})`);
                    console.log(`Error Body: ${body}`);
                    resolve(false);
                }
            });
        });

        req.on('error', (e) => {
            console.log(`❌ Network Error: ${e.message}`);
            resolve(false);
        });

        req.write(data);
        req.end();
    });
}

async function run() {
    // 1. Test o3 (Verified Available)
    await testModel("o3");
    
    // 2. Test o3-mini (Alternative)
    await testModel("o3-mini");
}

run();
