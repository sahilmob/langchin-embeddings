import "dotenv/config";
import { readFileSync } from "fs";
import { createClient } from "@supabase/supabase-js";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";

try {
  const text = readFileSync("server/scrimba.txt", "utf8");
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });

  const output = await splitter.createDocuments([text]);
  const supabaseurl = process.env.supabaseurl;
  const supabaseApiKey = process.env.supabaseApiKey;
  const openAIApiKey = process.env.openAIApiKey;

  const client = createClient(supabaseurl, supabaseApiKey);

  await SupabaseVectorStore.fromDocuments(
    output,
    new OpenAIEmbeddings({ openAIApiKey }),
    { client, tableName: "documents" }
  );
  console.log("done");
} catch (e) {
  console.error(e);
}
