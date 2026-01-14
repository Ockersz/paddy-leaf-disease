import React, { useEffect, useMemo, useRef, useState } from "react";
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
  LinearProgress,
  ToggleButton,
  ToggleButtonGroup,
} from "@mui/material";
import Chip from "@mui/material/Chip";
import {
  Brightness4,
  Brightness7,
  Send,
  RestartAlt,
} from "@mui/icons-material";
import AddIcon from "@mui/icons-material/Add";
import type { PaletteMode } from "@mui/material";

type ChatRole = "user" | "assistant";
type Language = "en" | "si";

interface ImagePrediction {
  filename: string;
  top_class: string;
  top_confidence: number; // 0–1
}

interface PredictionMeta {
  disease: string;
  overallConfidence: number; // 0–1
  imagePredictions: ImagePrediction[];
}

interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  predictions?: PredictionMeta;
}

interface PaddyChatProps {
  mode: PaletteMode;
  toggleMode: () => void;
}

const API_URL = "http://127.0.0.1:5000/api/chat";

const createSessionId = () =>
  `sess_${Math.random().toString(36).slice(2)}_${Date.now().toString(36)}`;

// ---------- UI strings (EN + SI) ----------
const STRINGS: Record<Language, any> = {
  en: {
    title: "Paddy Disease Assistant",
    subtitle:
      "Upload up to 5 leaf images and describe symptoms. I'll keep this session's history for follow-up questions.",
    session: "Session",
    newChat: "New chat",
    addImagesA11y: "Add images",
    imagesOnly: "[Images only]",
    thinking: "Thinking about your field…",
    waiting: "Waiting for reply...",
    placeholder:
      "Type your question or describe leaf symptoms. Press Enter to send, Shift+Enter for new line.",
    uploadLimit: "You can upload up to 5 images at a time.",
    errorGeneric:
      "Sorry, something went wrong while getting a reply. Please try again.",
    fallbackReach:
      "📡 I couldn't reach the backend right now. This is where I would give you a detailed answer based on the images and symptoms you sent.\n\n",
    youSaid: 'You said: "',
    initialAssistant:
      "Hi, I'm your paddy disease assistant. You can describe leaf symptoms or ask about a disease like blast, brown spot, hispa, dead heart or tungro. I'll explain symptoms, causes and management, and I can refine treatments if you tell me the weather and crop stage.",
    modelDiagnosis: "Model diagnosis",
    mostLikely: "Most likely disease",
    perImage: "Per-image confidence",
    langEN: "EN",
    langSI: "සිංහල",
  },
  si: {
    title: "වී රෝග සහායකයා",
    subtitle:
      "පත්‍ර රූප 5ක් දක්වා උඩුගත කර ලක්ෂණ විස්තර කරන්න. පසුව අසන ප්‍රශ්න සඳහා මෙම සැසියේ පසුබැසීම මතක තබාගන්නවා.",
    session: "සැසිය",
    newChat: "නව කතාබසක්",
    addImagesA11y: "රූප එක් කරන්න",
    imagesOnly: "[රූප පමණයි]",
    thinking: "ඔබගේ වගාව පිළිබඳ සිතා බලමින්…",
    waiting: "පිළිතුර රැගෙන එමින්...",
    placeholder:
      "ප්‍රශ්නය ටයිප් කරන්න හෝ පත්‍ර ලක්ෂණ විස්තර කරන්න. යැවීමට Enter, නව පේළියකට Shift+Enter.",
    uploadLimit: "එකවර රූප 5ක් දක්වා පමණක් උඩුගත කළ හැක.",
    errorGeneric:
      "කණගාටුයි, පිළිතුර ලබාගැනීමේදී ගැටලුවක් උදා විය. නැවත උත්සාහ කරන්න.",
    fallbackReach:
      "📡 මේ මොහොතේ backend එකට සම්බන්ධ වීමට නොහැකි විය. ඔබ යැවූ රූප සහ ලක්ෂණ මත පදනම්ව මෙතැන විස්තරාත්මක පිළිතුරක් ලබාදෙන්නෙමි.\n\n",
    youSaid: 'ඔබ කිව්වේ: "',
    initialAssistant:
      "හෙලෝ, මම ඔබගේ වී රෝග සහායකයා. පත්‍ර ලක්ෂණ විස්තර කරන්න හෝ වී බ්ලෑස්ට්, දම් ලප රෝගය, හිස්පා කීට, මෘත හද, ටන්ග්රෝ රෝගය වැනි රෝග ගැන අහන්න. ලක්ෂණ, හේතු සහ කළමනාකරණය කියලා දෙන්නම්. කාලගුණය සහ වගා අදියර කියලා දුන්නොත් ප්‍රතිකාර තවත් නිවැරදි කරලා දෙන්න පුළුවන්.",
    modelDiagnosis: "මොඩල් නිගමනය",
    mostLikely: "වැඩි ඉඩකඩ ඇත්තේ",
    perImage: "රූප අනුව විශ්වාසය",
    langEN: "EN",
    langSI: "සිංහල",
  },
};

const PaddyChat: React.FC<PaddyChatProps> = ({ mode, toggleMode }) => {
  const theme = useTheme();

  // ✅ Language state
  const [lang, setLang] = useState<Language>("en");
  const t = useMemo(() => STRINGS[lang], [lang]);

  const [messages, setMessages] = useState<ChatMessage[]>([
    { id: "m-0", role: "assistant", content: STRINGS.en.initialAssistant },
  ]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [images, setImages] = useState<File[]>([]);

  const sessionIdRef = useRef<string>(createSessionId());
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const canSend = !isSending && (input.trim().length > 0 || images.length > 0);

  // Keep initial assistant text aligned with selected language on fresh loads
  useEffect(() => {
    // If this is still the very first message, update it when language changes
    setMessages((prev) => {
      if (
        prev.length === 1 &&
        prev[0].id === "m-0" &&
        prev[0].role === "assistant"
      ) {
        return [{ ...prev[0], content: t.initialAssistant }];
      }
      return prev;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lang]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages.length]);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    const fileArray = Array.from(files);
    setImages((prev) => {
      const merged = [...prev, ...fileArray];
      if (merged.length > 5) setError(t.uploadLimit);
      return merged.slice(0, 5);
    });

    e.target.value = "";
  };

  const handleRemoveImage = (index: number) => {
    setImages((prev) => prev.filter((_, i) => i !== index));
  };

  // ---- API call (multipart) ----
  const sendToBackend = async (
    text: string,
    history: ChatMessage[],
    imageFiles: File[],
    language: Language
  ) => {
    const formData = new FormData();

    formData.append("session_id", sessionIdRef.current);
    formData.append("message", text);
    formData.append("language", language); // ✅ tell backend which language to reply in

    formData.append(
      "history",
      JSON.stringify(history.map((m) => ({ role: m.role, content: m.content })))
    );

    imageFiles.forEach((file) => {
      formData.append("images", file);
    });

    const res = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();

    if (data.session_id && typeof data.session_id === "string") {
      sessionIdRef.current = data.session_id;
    }

    return data;
  };

  const resetSession = (newLang?: Language) => {
    sessionIdRef.current = createSessionId();
    setMessages([
      {
        id: "m-0",
        role: "assistant",
        content: (newLang ? STRINGS[newLang] : STRINGS[lang]).initialAssistant,
      },
    ]);
    setInput("");
    setImages([]);
    setError(null);
    setIsSending(false);
    if (scrollRef.current) scrollRef.current.scrollTop = 0;
  };

  const handleSend = async () => {
    if (!canSend) return;

    const text = input.trim();
    setInput("");
    setError(null);

    const userTextSummary = text || (images.length > 0 ? t.imagesOnly : "");

    const userMessage: ChatMessage = {
      id: `m-${Date.now()}-user`,
      role: "user",
      content: userTextSummary,
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsSending(true);

    try {
      const historySnapshot = [...messages, userMessage];

      let replyText: string;
      let predictionMeta: PredictionMeta | undefined = undefined;

      try {
        const data = await sendToBackend(text, historySnapshot, images, lang);

        replyText = data.reply || "—";

        if (images.length > 0 && data.debug) {
          const dbg = data.debug;
          const disease = dbg.cnn_disease || data.disease_name || "Unknown";
          const overall =
            typeof dbg.cnn_confidence === "number" ? dbg.cnn_confidence : 0;

          const imagePredictions: ImagePrediction[] = Array.isArray(
            dbg.image_predictions
          )
            ? dbg.image_predictions.map((p: any) => ({
                filename: p.filename || "Image",
                top_class: p.top_class || disease,
                top_confidence:
                  typeof p.top_confidence === "number" ? p.top_confidence : 0,
              }))
            : [];

          predictionMeta = {
            disease,
            overallConfidence: overall,
            imagePredictions,
          };
        }
      } catch (err) {
        console.warn("Backend error, falling back:", err);
        replyText = t.fallbackReach + (text ? `${t.youSaid}${text}"` : "");
      }

      const botMessage: ChatMessage = {
        id: `m-${Date.now()}-assistant`,
        role: "assistant",
        content: replyText,
        predictions: predictionMeta,
      };

      setMessages((prev) => [...prev, botMessage]);
      setImages([]);
    } catch (e) {
      console.error(e);
      setError(t.errorGeneric);
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

  // 🔄 New chat
  const handleNewChat = () => resetSession();

  // 🌐 Language change → new session (required)
  const handleLanguageChange = (_: any, next: Language | null) => {
    if (!next || next === lang) return;
    setLang(next);
    resetSession(next); // ✅ new session + clear chat when language changes
  };

  const backgroundGradient =
    mode === "light"
      ? "linear-gradient(135deg, #e8f5e9, #e3f2fd)"
      : "linear-gradient(135deg, #020617, #111827)";

  const confidenceChipColor = (p: number): "success" | "warning" | "error" => {
    if (p >= 0.75) return "success";
    if (p >= 0.6) return "warning";
    return "error";
  };

  const renderPredictionCard = (pred: PredictionMeta) => {
    const overallPercent = (pred.overallConfidence * 100).toFixed(1);
    const chipColor = confidenceChipColor(pred.overallConfidence);

    return (
      <Box
        sx={{
          mt: 1.5,
          p: 1.5,
          borderRadius: 1.5,
          border: `1px solid ${theme.palette.divider}`,
          bgcolor:
            theme.palette.mode === "light"
              ? "rgba(229, 246, 253, 0.8)"
              : "rgba(15, 23, 42, 0.95)",
        }}
      >
        <Stack spacing={1.25}>
          <Stack
            direction="row"
            alignItems="center"
            justifyContent="space-between"
            spacing={1}
          >
            <Typography
              variant="caption"
              sx={{ textTransform: "uppercase", letterSpacing: 0.5 }}
              color="text.secondary"
            >
              {t.modelDiagnosis}
            </Typography>
            <Chip size="small" color={chipColor} label={`${overallPercent}%`} />
          </Stack>

          <Typography variant="body2">
            {t.mostLikely}:{" "}
            <Box component="span" fontWeight={600}>
              {pred.disease}
            </Box>
          </Typography>

          {pred.imagePredictions.length > 0 && (
            <Box>
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ display: "block", mb: 0.5 }}
              >
                {t.perImage}
              </Typography>
              <Stack spacing={0.75}>
                {pred.imagePredictions.map((p, idx) => {
                  const percent = (p.top_confidence * 100).toFixed(1);
                  const shortName =
                    p.filename.length > 18
                      ? p.filename.slice(0, 15) + "..."
                      : p.filename;

                  return (
                    <Box key={`${p.filename}-${idx}`}>
                      <Stack
                        direction="row"
                        justifyContent="space-between"
                        alignItems="center"
                        sx={{ mb: 0.25 }}
                      >
                        <Typography variant="caption" noWrap>
                          {shortName}
                        </Typography>
                        <Typography
                          variant="caption"
                          sx={{ fontFamily: "monospace" }}
                        >
                          {percent}% {p.top_class}
                        </Typography>
                      </Stack>
                      <LinearProgress
                        variant="determinate"
                        value={Math.max(
                          0,
                          Math.min(100, p.top_confidence * 100)
                        )}
                        sx={{ height: 6, borderRadius: 999 }}
                      />
                    </Box>
                  );
                })}
              </Stack>
            </Box>
          )}
        </Stack>
      </Box>
    );
  };

  const renderMessage = (msg: ChatMessage) => {
    const isUser = msg.role === "user";
    const isAssistant = msg.role === "assistant";

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
          {isAssistant &&
            msg.predictions &&
            renderPredictionCard(msg.predictions)}
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
                {t.title}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {t.subtitle}
              </Typography>
            </Box>
          </Stack>

          <Stack spacing={0.75} alignItems="flex-end">
            <Typography variant="caption" color="text.secondary">
              {t.session}: {sessionIdRef.current}
            </Typography>

            <Stack direction="row" spacing={1} alignItems="center">
              {/* 🌐 Language toggle */}
              <ToggleButtonGroup
                size="small"
                exclusive
                value={lang}
                onChange={handleLanguageChange}
                aria-label="language"
              >
                <ToggleButton value="en" aria-label="English">
                  {STRINGS.en.langEN}
                </ToggleButton>
                <ToggleButton value="si" aria-label="Sinhala">
                  {STRINGS.si.langSI}
                </ToggleButton>
              </ToggleButtonGroup>

              {/* New chat */}
              <Button
                variant="outlined"
                size="small"
                startIcon={<RestartAlt fontSize="small" />}
                onClick={handleNewChat}
              >
                {t.newChat}
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
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
              {t.thinking}
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

        {/* Input + images */}
        <Box
          sx={{
            borderTop: `1px solid ${theme.palette.divider}`,
            px: 2.5,
            py: 1.5,
          }}
        >
          {images.length > 0 && (
            <Stack direction="row" spacing={1} sx={{ mb: 1, flexWrap: "wrap" }}>
              {images.map((file, idx) => (
                <Chip
                  key={`${file.name}-${idx}`}
                  label={file.name}
                  size="small"
                  onDelete={() => handleRemoveImage(idx)}
                  sx={{ maxWidth: 200 }}
                />
              ))}
            </Stack>
          )}

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
            <Button
              variant="outlined"
              component="label"
              size="small"
              aria-label={t.addImagesA11y}
              sx={{ alignSelf: "stretch", whiteSpace: "nowrap" }}
            >
              {images.length > 0 ? `${images.length}/5` : <AddIcon />}
              <input
                type="file"
                accept="image/*"
                multiple
                hidden
                onChange={handleImageChange}
              />
            </Button>

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
              placeholder={isSending ? t.waiting : t.placeholder}
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
