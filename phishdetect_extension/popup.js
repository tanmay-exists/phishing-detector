// popup.js

document.getElementById("checkEmails").addEventListener("click", () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        // Send a message to the content script in the active tab
        chrome.tabs.sendMessage(tabs[0].id, { action: "recheckEmails" }, (response) => {
            if (chrome.runtime.lastError) {
                // This will catch errors if the content script isn't loaded
                alert("Could not connect to the page. Please refresh Gmail and try again.");
                console.error(chrome.runtime.lastError.message);
            } else {
                alert("✅ Re-check command sent! Highlights will update shortly.");
                console.log(response.status);
            }
        });
    });
});