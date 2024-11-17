// Get modal elements
const modals = [
    document.getElementById("dataModal1"),
    document.getElementById("dataModal2"),
    document.getElementById("dataModal3"),
    document.getElementById("dataModal4")
];

// Get button elements to open modals
const openButtons = [
    document.getElementById("openModal1"),
    document.getElementById("openModal2"),
    document.getElementById("openModal3"),
    document.getElementById("openModal4")
];

// Get close elements for modals
const closeButtons = [
    document.getElementById("closeModal1"),
    document.getElementById("closeModal2"),
    document.getElementById("closeModal3"),
    document.getElementById("closeModal4")
];

// Function to open modals
openButtons.forEach((button, index) => {
    button.onclick = function() {
        modals[index].style.display = "block";
    };
});

// Function to close modals
closeButtons.forEach((button, index) => {
    button.onclick = function() {
        modals[index].style.display = "none";
    };
});

// Close modals when clicking outside of them
window.onclick = function(event) {
    modals.forEach(modal => {
        if (event.target === modal) {
            modal.style.display = "none";
        }
    });
};
