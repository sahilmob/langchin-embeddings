import List from "@mui/material/List";
import Avatar from "@mui/material/Avatar";
import Divider from "@mui/material/Divider";
import ListItem from "@mui/material/ListItem";
import TextField from "@mui/material/TextField";
import Grid from "@mui/material/Unstable_Grid2";
import { PromptTemplate } from "langchain/prompts";
import { useRef, useState, Fragment } from "react";
import ListItemText from "@mui/material/ListItemText";
import { ChatOpenAI } from "langchain/chat_models/openai";
import ListItemAvatar from "@mui/material/ListItemAvatar";
import AccountCircleIcon from "@mui/icons-material/AccountCircle";
import SupportAgentIcon from "@mui/icons-material/SupportAgent";

const openAIApiKey = process.env.REACT_APP_openAIApiKey;
const llm = new ChatOpenAI({
  openAIApiKey,
});

const template =
  "Given a question, convert it to a standalone question. question: {question} standalone question:";

const prompt = PromptTemplate.fromTemplate(template);

const chain = prompt.pipe(llm);

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
