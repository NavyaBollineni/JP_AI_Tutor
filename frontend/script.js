
const API_BASE = "";

const welcome = document.getElementById("welcome");
const messages = document.getElementById("messages");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const sendButton = document.getElementById("sendButton");
const languageSelect = document.getElementById("languageSelect");
const fileInput = document.getElementById("fileInput");
const attachButton = document.getElementById("attachButton");
const sidebarUpload = document.getElementById("sidebarUpload");
const uploadStatus = document.getElementById("uploadStatus");
const uploadName = document.getElementById("uploadName");
const removeUpload = document.getElementById("removeUpload");
const toast = document.getElementById("toast");
const sidebar = document.getElementById("sidebar");

let selectedFile = null;
let isBusy = false;
let toastTimer = null;

// Create a unique ID for each conversation.
function createConversationId() {
    if (
        window.crypto &&
        typeof window.crypto.randomUUID === "function"
    ) {
        return window.crypto.randomUUID();
    }

    return (
        "chat-" +
        Date.now() +
        "-" +
        Math.random().toString(36).slice(2)
    );
}

// All messages in this conversation use the same ID.
let conversationId = createConversationId();

// Display a notification.
function showToast(message) {
    toast.textContent = message;
    toast.classList.add("visible");

    clearTimeout(toastTimer);

    toastTimer = setTimeout(() => {
        toast.classList.remove("visible");
    }, 3200);
}

// Sidebar controls.
function closeSidebar() {
    sidebar.classList.remove("open");
}

document.getElementById("menuButton").addEventListener(
    "click",
    () => sidebar.classList.toggle("open")
);

document.getElementById("closeSidebar").addEventListener(
    "click",
    closeSidebar
);

// Start a new conversation with a fresh ID.
function startNewChat() {
    if (isBusy) {
        showToast(
            "Please wait for the current answer to finish."
        );
        return;
    }

    conversationId = createConversationId();

    messages.replaceChildren();

    welcome.hidden = false;
    questionInput.value = "";
    questionInput.style.height = "auto";

    closeSidebar();
    questionInput.focus();
}

document.getElementById("newChat").addEventListener(
    "click",
    startNewChat
);

// Suggested prompts.
document.querySelectorAll("[data-prompt]").forEach(
    button => {
        button.addEventListener("click", () => {
            questionInput.value = button.dataset.prompt;
            questionInput.focus();

            questionInput.dispatchEvent(
                new Event("input")
            );

            if (window.innerWidth <= 760) {
                closeSidebar();
            }
        });
    }
);

// Automatically resize the question field.
questionInput.addEventListener("input", () => {
    questionInput.style.height = "auto";

    questionInput.style.height =
        Math.min(questionInput.scrollHeight, 140) + "px";
});

// Enter sends; Shift + Enter adds a new line.
questionInput.addEventListener("keydown", event => {
    if (
        event.key === "Enter" &&
        !event.shiftKey
    ) {
        event.preventDefault();
        chatForm.requestSubmit();
    }
});

// Document upload controls.
function openFilePicker() {
    fileInput.click();
}

attachButton.addEventListener("click", openFilePicker);

sidebarUpload.addEventListener("click", () => {
    closeSidebar();
    openFilePicker();
});

// Select a supported document.
fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];

    if (!file) {
        return;
    }

    const allowedExtensions = [
        ".pdf",
        ".docx",
        ".xlsx",
        ".xls",
        ".txt",
        ".csv"
    ];

    const extension =
        "." + file.name.split(".").pop().toLowerCase();

    if (!allowedExtensions.includes(extension)) {
        showToast("Please select a supported document.");
        fileInput.value = "";
        selectedFile = null;
        return;
    }

    selectedFile = file;
    uploadName.textContent = file.name;
    uploadStatus.hidden = false;

    showToast(
        "Document selected. Ask a question to upload it."
    );
});

// Remove the selected document.
removeUpload.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    uploadStatus.hidden = true;
    uploadName.textContent = "";
});

// Safely display a chat message.
function addMessage(role, text) {
    welcome.hidden = true;

    const message = document.createElement("article");
    message.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "You" : "日";

    const body = document.createElement("div");
    body.className = "message-body";

    const name = document.createElement("div");
    name.className = "message-name";
    name.textContent =
        role === "user" ? "You" : "JP AI Tutor";

    const content = document.createElement("div");
    content.className = "message-text";
    content.textContent = text;

    body.append(name, content);
    message.append(avatar, body);
    messages.appendChild(message);

    const conversation =
        document.getElementById("conversation");

    conversation.scrollTop = conversation.scrollHeight;

    return { message, content };
}

// Display a typing indicator.
function showTyping() {
    welcome.hidden = true;

    const message = document.createElement("article");
    message.className = "message assistant";
    message.id = "typingIndicator";

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "日";

    const body = document.createElement("div");
    body.className = "message-body";

    const name = document.createElement("div");
    name.className = "message-name";
    name.textContent = "JP AI Tutor";

    const typing = document.createElement("div");
    typing.className = "typing";
    typing.textContent = "Thinking";

    const dots = document.createElement("span");
    dots.textContent = " ...";
    typing.appendChild(dots);

    body.append(name, typing);
    message.append(avatar, body);
    messages.appendChild(message);

    const conversation =
        document.getElementById("conversation");

    conversation.scrollTop = conversation.scrollHeight;
}

function hideTyping() {
    document.getElementById("typingIndicator")?.remove();
}

// Read useful error details from FastAPI.
async function readError(response) {
    const error = await response.json().catch(() => ({}));

    if (typeof error.detail === "string") {
        return error.detail;
    }

    if (Array.isArray(error.detail)) {
        return error.detail
            .map(item => {
                const location = Array.isArray(item.loc)
                    ? item.loc.join(" → ")
                    : "Request";

                return `${location}: ${item.msg || "Invalid value"}`;
            })
            .join("\n");
    }

    if (error.message) {
        return error.message;
    }

    return `Request failed (${response.status}).`;
}

// Upload the selected document to FastAPI.
async function uploadSelectedFile() {
    if (!selectedFile) {
        return null;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData
    });

    if (!response.ok) {
        throw new Error(await readError(response));
    }

    return await response.json();
}

// Send the question, selected language, and conversation ID.
async function sendQuestion(question) {
    const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            question: question,
            language: languageSelect.value,
            conversation_id: conversationId
        })
    });

    if (!response.ok) {
        throw new Error(await readError(response));
    }

    return await response.json();
}

// Handle chat submission.
chatForm.addEventListener("submit", async event => {
    event.preventDefault();

    if (isBusy) {
        return;
    }

    const question = questionInput.value.trim();

    if (!question && !selectedFile) {
        showToast(
            "Ask a question or select a document first."
        );
        questionInput.focus();
        return;
    }

    if (!question && selectedFile) {
        showToast(
            "Document selected. Enter a question about it."
        );
        questionInput.focus();
        return;
    }

    isBusy = true;
    sendButton.disabled = true;

    addMessage("user", question);

    questionInput.value = "";
    questionInput.style.height = "auto";

    showTyping();

    try {
        // Upload the selected document first.
        if (selectedFile) {
            await uploadSelectedFile();

            selectedFile = null;
            fileInput.value = "";
            uploadStatus.hidden = true;
            uploadName.textContent = "";

            showToast("Document uploaded successfully.");
        }

        // Send the question to the backend.
        const result = await sendQuestion(question);

        hideTyping();

        if (
            result.status &&
            result.status !== "success"
        ) {
            throw new Error(
                result.message || "The tutor could not answer."
            );
        }

        const answer =
            result.answer ??
            result.response ??
            result.message;

        if (
            typeof answer !== "string" ||
            !answer.trim()
        ) {
            throw new Error(
                "The server returned an empty answer."
            );
        }

        addMessage("assistant", answer);

    } catch (error) {
        hideTyping();

        const details =
            error instanceof TypeError
                ? "Could not connect to the backend. Check that Docker is running at http://localhost:8000."
                : error.message;

        const errorMessage = addMessage(
            "assistant",
            `Sorry, something went wrong.\n${details}`
        );

        errorMessage.content.classList.add("error");

    } finally {
        isBusy = false;
        sendButton.disabled = false;
        questionInput.focus();
    }
});
