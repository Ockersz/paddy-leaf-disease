import React, { useEffect, useRef, useState } from "react";
import {
  Avatar,
  Box,
  IconButton,
  Paper,
  Stack,
  TextField,
  Typography,
  Button,
  useTheme,
} from "@mui/material";
import { Brightness4, Brightness7, Send, RestartAlt } from "@mui/icons-material";
import type { PaletteMode } from "@mui/material";

type ChatRole = "user" | "assistant";

interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
}

interface PaddyChatProps {
  mode: PaletteMode;
  toggleMode: () => void;
}

const API_URL = "http://127.0.0.1:8000/api/chat";

const createSessionId = () =>
  `sess_${Math.random().toString(36).slice(2)}_${Date.now().toString(36)}`;

const initialAssistantMessage =
  "Hi, I’m your paddy disease assistant. You can describe leaf symptoms or ask about a disease like blast, brown spot, hispa, dead heart or tungro. I’ll explain symptoms, causes and management, and I can refine treatments if you tell me the weather and crop stage.";

const PaddyChat: React.FC<PaddyChatProps> = ({ mode, toggleMode }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { id: "m-0", role: "assistant", content: initialAssistantMessage },
  ]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sessionIdRef = useRef<string>(createSessionId());
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const theme = useTheme();

  const canSend = input.trim().length > 0 && !isSending;

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages.length]);

  // ---- API call ----
  const sendToBackend = async (text: string, history: ChatMessage[]) => {
    const payload = {
      session_id: sessionIdRef.current,
      message: text,
      history: history.map((m) => ({ role: m.role, content: m.content })),
    };

    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const data = await res.json();

    if (data.session_id && typeof data.session_id === "string") {
      sessionIdRef.current = data.session_id;
    }

    return data; // return full response, not just reply text
  };

  const handleSend = async () => {
    if (!canSend) return;
    const text = input.trim();
    setInput("");
    setError(null);

    const userMessage: ChatMessage = {
      id: `m-${Date.now()}-user`,
      role: "user",
      content: text,
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsSending(true);

    try {
      const historySnapshot = [...messages, userMessage];

      let replyText: string;

      try {
        const data = await sendToBackend(text, historySnapshot);
        replyText = data.reply || "I couldn’t generate a proper reply.";
        // (optionally) inspect data.awaiting_refinement, data.debug, etc. here
      } catch (err) {
        console.warn("Backend error, falling back:", err);
        replyText =
          "📡 I couldn’t reach the backend right now. This is where I would give you a detailed answer " +
          "based on your symptoms and previous messages.\n\n" +
          `You said: "${text}"`;
      }

      const botMessage: ChatMessage = {
        id: `m-${Date.now()}-assistant`,
        role: "assistant",
        content: replyText,
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (e: any) {
      console.error(e);
      setError(
        "Sorry, something went wrong while getting a reply. Please try again."
      );
    } finally {
      setIsSending(false);
    }
  };


  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // 🔄 NEW CHAT HANDLER
  const handleNewChat = () => {
    sessionIdRef.current = createSessionId(); // new session id
    setMessages([
      { id: "m-0", role: "assistant", content: initialAssistantMessage },
    ]);
    setInput("");
    setError(null);
    setIsSending(false);
    // optional: scroll to top
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
  };

  const backgroundGradient =
    mode === "light"
      ? "linear-gradient(135deg, #e8f5e9, #e3f2fd)"
      : "linear-gradient(135deg, #020617, #111827)";

  const renderMessage = (msg: ChatMessage) => {
    const isUser = msg.role === "user";
    return (
      <Stack
        key={msg.id}
        direction="row"
        spacing={1.5}
        justifyContent={isUser ? "flex-end" : "flex-start"}
        alignItems="flex-start"
      >
        {!isUser && (
          <Avatar
            sx={{
              bgcolor: theme.palette.success.light,
              color: theme.palette.success.contrastText,
              fontSize: 12,
            }}
          >
            AI
          </Avatar>
        )}

        <Box
          sx={{
            maxWidth: { xs: "80%", md: "70%" },
            bgcolor: isUser
              ? theme.palette.primary.main
              : theme.palette.mode === "light"
                ? theme.palette.background.paper
                : "#020617",
            color: isUser
              ? theme.palette.primary.contrastText
              : theme.palette.text.primary,
            borderRadius: 1,
            px: 2,
            py: 1.2,
            boxShadow: 1,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
          }}
        >
          <Typography variant="body2">{msg.content}</Typography>
        </Box>

        {isUser && (
          <Avatar
            sx={{
              bgcolor: theme.palette.primary.light,
              color: theme.palette.primary.contrastText,
              fontSize: 12,
            }}
          >
            You
          </Avatar>
        )}
      </Stack>
    );
  };

  return (
    <Box
      sx={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        p: { xs: 1.5, sm: 2 },
        backgroundImage: backgroundGradient,
      }}
    >
      <Paper
        elevation={8}
        sx={{
          width: "100%",
          maxWidth: 960,
          height: "90vh",
          display: "flex",
          flexDirection: "column",
          borderRadius: 2,
          overflow: "hidden",
        }}
      >
        {/* Header */}
        <Box
          sx={{
            px: 1.5,
            py: 1.5,
            borderBottom: `1px solid ${theme.palette.divider}`,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 1,
          }}
        >
          <Stack direction="row" spacing={1.5} alignItems="center">
            <Avatar
              sx={{
                bgcolor: theme.palette.success.light,
                color: theme.palette.success.contrastText,
              }}
            >
              AI
            </Avatar>
            <Box>
              <Typography variant="subtitle1" fontWeight={600}>
                Paddy Disease Assistant
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Describe leaf symptoms or ask follow-up questions. I’ll use this
                session’s history to keep context.
              </Typography>
            </Box>
          </Stack>

          <Stack spacing={0.5} alignItems="flex-end">
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ maxWidth: 200 }}
            >
              Session: {sessionIdRef.current}
            </Typography>

            <Stack direction="row" spacing={1}>
              {/* New chat button */}
              <Button
                variant="outlined"
                size="small"
                startIcon={<RestartAlt fontSize="small" />}
                onClick={handleNewChat}
              >
                New chat
              </Button>

              {/* Theme toggle */}
              <IconButton
                size="small"
                onClick={toggleMode}
                sx={{
                  borderRadius: 999,
                  bgcolor:
                    mode === "light"
                      ? "rgba(0,0,0,0.04)"
                      : "rgba(255,255,255,0.08)",
                }}
              >
                {mode === "light" ? (
                  <Brightness4 fontSize="small" />
                ) : (
                  <Brightness7 fontSize="small" />
                )}
              </IconButton>
            </Stack>
          </Stack>
        </Box>

        {/* Messages */}
        <Box
          ref={scrollRef}
          sx={{
            flex: 1,
            overflowY: "auto",
            px: 2.5,
            py: 2,
            display: "flex",
            flexDirection: "column",
            gap: 1.5,
          }}
        >
          {messages.map(renderMessage)}
          {isSending && (
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ mt: 1 }}
            >
              Thinking about your field…
            </Typography>
          )}
        </Box>

        {/* Error */}
        {error && (
          <Box sx={{ px: 2.5, pb: 0.5 }}>
            <Typography variant="caption" color="error">
              {error}
            </Typography>
          </Box>
        )}

        {/* Input */}
        <Box
          sx={{
            borderTop: `1px solid ${theme.palette.divider}`,
            px: 2.5,
            py: 1.5,
          }}
        >
          <Stack
            direction="row"
            spacing={1.5}
            alignItems="flex-end"
            component="form"
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
          >
            <TextField
              fullWidth
              multiline
              minRows={2}
              maxRows={4}
              size="small"
              variant="outlined"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                isSending
                  ? "Waiting for reply..."
                  : "Type your question or describe leaf symptoms. Press Enter to send, Shift+Enter for new line."
              }
            />
            <Button
              type="submit"
              variant="contained"
              color="primary"
              disabled={!canSend}
              endIcon={<Send fontSize="small" />}
              sx={{ px: 3, py: 1.2, alignSelf: "stretch" }}
            >
              Send
            </Button>
          </Stack>
        </Box>
      </Paper>
    </Box>
  );
};

export default PaddyChat;
