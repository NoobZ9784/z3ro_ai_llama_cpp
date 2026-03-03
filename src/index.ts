import path from "path";
import {
  getLlama,
  LlamaChatSession,
  LlamaModel
} from "node-llama-cpp";

process.env.CUDA_VISIBLE_DEVICES = "0";
process.env.VK_ICD_FILENAMES =
  "C:\\Windows\\System32\\nv-vk64.json";

const prompt = "What is walrus operator in python";

let session: LlamaChatSession | null = null;

async function getLLMSession() {
  if (session) return session;

  const llama = await getLlama({
    gpu: false
  });

  const modelPath = path.join(
    process.cwd(),
    "Models/gemma-3-4b-it-Q2_K.gguf"
    // "Models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
  );

  const model: LlamaModel = await llama.loadModel({
    modelPath
  });

  const context = await model.createContext({
    contextSize: 4096
  });

  session = new LlamaChatSession({
    contextSequence: context.getSequence(),
    systemPrompt: `
You are a local AI assistant name Z3RO AI.
Be concise, accurate, and helpful.
`
  });

  return session;
}

(async () => {
  console.log(process.cwd());
  let i = 0;

  const ms = getLLMSession();
  (await ms).prompt(prompt, {
    onTextChunk(chunk) {
      console.log(chunk)
    }
  });
})()
