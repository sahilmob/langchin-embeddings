import List from "@mui/material/List";
import Avatar from "@mui/material/Avatar";
import Divider from "@mui/material/Divider";
import ListItem from "@mui/material/ListItem";
import TextField from "@mui/material/TextField";
import Grid from "@mui/material/Unstable_Grid2";
import { PromptTemplate } from "langchain/prompts";
import { useRef, useState, Fragment } from "react";
import { createClient } from "@supabase/supabase-js";
import { OpenAIEmbeddings } from "@langchain/openai";
import ListItemText from "@mui/material/ListItemText";
import { ChatOpenAI } from "langchain/chat_models/openai";
import ListItemAvatar from "@mui/material/ListItemAvatar";
import SupportAgentIcon from "@mui/icons-material/SupportAgent";
import AccountCircleIcon from "@mui/icons-material/AccountCircle";
import { StringOutputParser } from "langchain/schema/output_parser";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";

const openAIApiKey = process.env.REACT_APP_OPENAI_API_KEY;
const embddings = new OpenAIEmbeddings({
  openAIApiKey,
});
const sbApiKey = process.env.REACT_APP_SUPABASE_API_KEY;
const sbUrl = process.env.REACT_APP_SUPABASE_URL;
const client = createClient(sbUrl!, sbApiKey!);

const vectorStore = new SupabaseVectorStore(embddings, {
  client,
  tableName: "documents",
  queryName: "match_documents",
});

const retriever = vectorStore.asRetriever();

const llm = new ChatOpenAI({
  openAIApiKey,
});

const template =
  "Given a question, convert it to a standalone question. question: {question} standalone question:";
const answerTemplate = `You are a helpful and enthusiastic support bot who can answer any question
  about Scrimba based on the context provided. Try to find the answer in the context. If you can't find the answer, say "I'm sorry, I don't know the answer to that question. and direct questioner to email help@scrimba,com".
  Don't try to make up and answer. Always speak as if you are chatting to a friend.
  context: {context},
  question: {question},
  answer:
`;

const answerPrompt = PromptTemplate.fromTemplate(answerTemplate);
const prompt = PromptTemplate.fromTemplate(template);

const combineDocs = (docs: any[]) => {
  return docs.map((doc) => doc.pageContent).join("\n\n");
};

const chain = prompt
  .pipe(llm)
  .pipe(new StringOutputParser())
  .pipe(retriever)
  .pipe(combineDocs)
  .pipe(answerPrompt);

function App() {
  const [messages, setMessages] = useState<
    { message: string; role: "user" | "ai" }[]
  >([]);
  const inputRef = useRef<HTMLInputElement>(null);

  const submitUserInput = async () => {
    const input = inputRef.current;
    if (input) {
      setMessages((messages) => [
        ...messages,
        { message: input.value, role: "user" },
      ]);
      inputRef.current!.value = "";
      const response = await chain.invoke({
        question: input.value,
      });
    }
  };

  const keyDownHandler = (event: React.KeyboardEvent) => {
    if (event.key === "Enter" && event.shiftKey === false) {
      submitUserInput();
    }
  };

  return (
    <Grid
      container
      sx={{ flexDirection: "column", height: "100%", padding: 2 }}
    >
      <Grid
        xs={12}
        sx={{
          flexGrow: 1,
        }}
      >
        <List
          sx={{ width: "100%", maxWidth: 360, bgcolor: "background.paper" }}
        >
          {messages.map((m, i) => (
            <Fragment key={i}>
              <ListItem alignItems="flex-start">
                <ListItemAvatar>
                  <Avatar>
                    {m.role === "user" ? (
                      <AccountCircleIcon />
                    ) : (
                      <SupportAgentIcon />
                    )}
                  </Avatar>
                </ListItemAvatar>
                <ListItemText primary="You" secondary={m.message} />
              </ListItem>
              <Divider variant="inset" component="li" />
            </Fragment>
          ))}
        </List>
      </Grid>
      <Grid xs={12}>
        <TextField
          fullWidth
          multiline
          inputRef={inputRef}
          onKeyDown={keyDownHandler}
        />
      </Grid>
    </Grid>
  );
}

export default App;
