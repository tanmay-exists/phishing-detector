// content.js

const API_URL = "http://localhost:8000/predict";

async function checkEmail(emailDiv) {
    // Prevent re-checking an element if an API call is already in progress
    if (emailDiv.dataset.checking === "true") return;

    const text = emailDiv.innerText;
    if (!text) return;

    try {
        emailDiv.dataset.checking = "true"; // Mark as "in-progress"
        const resp = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });
        const data = await resp.json();

        // Reset style before applying new one
        emailDiv.style.border = "none";
        emailDiv.style.backgroundColor = "transparent";

        if (data.label === 1) {
            emailDiv.style.border = "3px solid red";
            emailDiv.style.backgroundColor = "#ffe6e6";
            emailDiv.title = `⚠️ Phishing Risk: ${data.prob.toFixed(2)}`;
        }
        emailDiv.dataset.checked = "true"; // Mark as "completed"
    } catch (err) {
        console.error("PhishDetect API error:", err);
        // Optionally add a visual indicator for error
        emailDiv.style.border = "3px solid orange";
        emailDiv.title = "Could not analyze this email.";
    } finally {
        emailDiv.dataset.checking = "false"; // Reset "in-progress" flag
    }
}

function runChecksOnPage() {
    console.log("PhishDetect: Running checks on all visible emails.");
    const emails = document.querySelectorAll('div[role="link"]');
    emails.forEach(email => {
        // For a manual re-check, we want to re-evaluate every email
        checkEmail(email);
    });
}

// 1. Listener for messages from the popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "recheckEmails") {
        runChecksOnPage();
        sendResponse({ status: "Emails are being re-checked." });
    }
    return true; // Indicates you wish to send a response asynchronously
});

// 2. Observer for new emails loading dynamically
const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        // Check if new nodes were added
        if (mutation.addedNodes.length) {
            const emails = document.querySelectorAll('div[role="link"]:not([data-checked="true"])');
            emails.forEach(email => {
                checkEmail(email);
            });
        }
    });
});

// Start observing the document body
observer.observe(document.body, { childList: true, subtree: true });

// Optional: Run once when the script is first injected
setTimeout(runChecksOnPage, 1000); // Give Gmail a second to finish loading