// Recupera gli elementi del DOM
const companyNameInput = document.getElementById("company-name");
const mainFieldInput = document.getElementById("main-field");
const submitButton = document.getElementById("submit-button");
const newPostPanel = document.getElementById("new-post-panel");
const postImage = document.getElementById("post-image");
const postText = document.getElementById("post-text");

// Funzione per abilitare o disabilitare il pulsante di invio
function toggleSubmitButton() {
    if (companyNameInput.value.trim() !== "" && mainFieldInput.value.trim() !== "") {
        submitButton.disabled = false;
    } else {
        submitButton.disabled = true;
    }
}

// Event listeners per rilevare le modifiche agli input
companyNameInput.addEventListener("input", toggleSubmitButton);
mainFieldInput.addEventListener("input", toggleSubmitButton);

// Inizializza il pulsante "Submit" come disabilitato
submitButton.disabled = true;

submitButton.addEventListener("click", function () {
    // Mostra il messaggio "Processing..."
    const processingMessage = document.getElementById("processing-message");
    processingMessage.style.display = "block";

    // Ottieni i valori di input
    const companyName = companyNameInput.value;
    const mainField = mainFieldInput.value;

    // Effettua la richiesta POST al server Flask
    fetch('/api/create_post', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ company_name: companyName, main_field: mainField })
    })
    .then(response => response.json())
    .then(data => {
        // Nascondi il messaggio "Processing..."
        processingMessage.style.display = "none";

        // Mostra il pannello del nuovo post
        newPostPanel.style.display = "block";
        postText.textContent = data.text;
        postImage.src = data.image_url;

        // Pulisci i campi di input
        companyNameInput.value = "";
        mainFieldInput.value = "";

        // Disabilita il pulsante "Submit" finché i campi non sono riempiti di nuovo
        submitButton.disabled = true;
    })
    .catch(error => {
        console.error('Errore durante la creazione del post:', error);
        processingMessage.textContent = "Errore durante la creazione del post";
    });
});