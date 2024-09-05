const container = document.getElementById("main-container")
const panel = document.getElementById("main-panel")
const companyNameInput = document.getElementById("company-name");
const mainFieldInput = document.getElementById("main-field");
const submitButton = document.getElementById("submit-button");
const newPostPanel = document.getElementById("new-post-panel");
const postImage = document.getElementById("post-image");
const postText = document.getElementById("post-text");
const advancedSettingButton = document.getElementById("advanced-settings")
const advancedSettingPanel = document.getElementById("advanced-panel")

function toggleSubmitButton() {
    if (companyNameInput.value.trim() !== "" && mainFieldInput.value.trim() !== "") {
        submitButton.disabled = false;
    } else {
        submitButton.disabled = true;
    }
}

function getImagePromptFields() {
    // Recupera i valori degli input per l'immagine
    const imageType = document.getElementById("image-type").value.trim();
    const subject = document.getElementById("subject").value.trim();
    const environment = document.getElementById("environment").value.trim();
    const light = document.getElementById("light").value.trim();
    const color = document.getElementById("color").value.trim();
    const pointView = document.getElementById("point-view").value.trim();
    const artStyle = document.getElementById("art-style").value.trim();
    const photoType = document.getElementById("photo-type").value.trim();

    // Costruisce un oggetto con tutti i campi dell'immagine
    return {
        image_type: imageType,
        subject: subject,
        environment: environment,
        light: light,
        color: color,
        point_view: pointView,
        art_style: artStyle,
        photo_type: photoType
    };
}

function getTextPromptFields() {
    // Recupera i valori degli input per il testo
    const goal = document.getElementById("goal").value.trim();
    const target = document.getElementById("target").value.trim();
    const style = document.getElementById("style").value.trim();
    const keywords = document.getElementById("keywords").value.trim();

    // Costruisce un oggetto con tutti i campi del testo
    return {
        goal: goal,
        target: target,
        style: style,
        keywords: keywords
    };
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
    const advancedSettingButtonValue = advancedSettingButton.checked;
    const imageFields = getImagePromptFields();
    const textFields = getTextPromptFields();

    const promptValues = {
        image_prompt: imageFields,
        text_prompt: textFields
    };


    // POST request at Flask server
    fetch('/api/create_post', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ company_name: companyName, main_field: mainField,
            setting_button: advancedSettingButtonValue, prompt_values: promptValues
            })
    })
    .then(response => response.json())
    .then(data => {
        processingMessage.style.display = "none";

        newPostPanel.style.display = "block";
        postText.textContent = data.text;
        postImage.src = data.image;

        companyNameInput.value = "";
        mainFieldInput.value = "";

        advancedSettingPanel.style.display = "none"
        advancedSettingButton.checked = false

        submitButton.disabled = true;
    })
    .catch(error => {
        console.error('Error during post creation:', error);
        processingMessage.textContent = "Error during post creation";
    });
});

advancedSettingButton.addEventListener("click", function() {

    // Case "Advanced Settings" not selected
    if(advancedSettingPanel.style.display != "flex") {
        advancedSettingPanel.style.display = "flex";
        container.style.width = "80%";
        container.style.maxWidth = "80%";
        panel.style.maxWidth = "100%";
    }
    else {
        advancedSettingPanel.style.display = "none";
        container.style.width = "100%";
        container.style.maxWidth = "350px";
        panel.style.maxWidth = "350px";
    }
})