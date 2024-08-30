const companyNameInput = document.getElementById("company-name");
const mainFieldInput = document.getElementById("main-field");
const submitButton = document.getElementById("submit-button");
const newPostPanel = document.getElementById("new-post-panel");
const postImage = document.getElementById("post-image");
const postText = document.getElementById("post-text");

function toggleSubmitButton() {
    if (companyNameInput.value.trim() !== "" && mainFieldInput.value.trim() !== "") {
        submitButton.disabled = false;
    } else {
        submitButton.disabled = true;
    }
}

companyNameInput.addEventListener("input", toggleSubmitButton);
mainFieldInput.addEventListener("input", toggleSubmitButton);

submitButton.disabled = true;

submitButton.addEventListener("click", function () {
    // Show "Processing..." message
    const processingMessage = document.getElementById("processing-message");
    processingMessage.style.display = "block";

    // Get input values
    const companyName = companyNameInput.value;
    const mainField = mainFieldInput.value;

    // POST request at Flask server
    fetch('/api/create_post', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ company_name: companyName, main_field: mainField })
    })
    .then(response => response.json())
    .then(data => {
        processingMessage.style.display = "none";

        newPostPanel.style.display = "block";
        postText.textContent = data.text;
        postImage.src = data.image_url;

        companyNameInput.value = "";
        mainFieldInput.value = "";

        submitButton.disabled = true;
    })
    .catch(error => {
        console.error('Error during post creation:', error);
        processingMessage.textContent = "Error during post creation";
    });
});