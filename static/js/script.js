document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    
    // Add form validation
    form.addEventListener('submit', function(e) {
        if (!validateForm()) {
            e.preventDefault();
        }
    });
    
    // Add input event listeners for real-time validation
    const inputs = form.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            validateField(this);
        });
    });
});

function validateForm() {
    const form = document.getElementById('predictionForm');
    const inputs = form.querySelectorAll('input[required], select[required]');
    let isValid = true;
    
    inputs.forEach(input => {
        if (!validateField(input)) {
            isValid = false;
        }
    });
    
    return isValid;
}

function validateField(field) {
    const value = field.value.trim();
    let isValid = true;
    
    // Remove existing error styling
    field.classList.remove('error');
    
    // Check if required field is empty
    if (field.hasAttribute('required') && !value) {
        isValid = false;
    }
    
    // Specific validations
    switch(field.name) {
        case 'age':
            if (value && (parseInt(value) < 16 || parseInt(value) > 30)) {
                isValid = false;
            }
            break;
        case 'study_hours':
            if (value && (parseInt(value) < 0 || parseInt(value) > 50)) {
                isValid = false;
            }
            break;
        case 'attendance_rate':
            if (value && (parseFloat(value) < 0 || parseFloat(value) > 100)) {
                isValid = false;
            }
            break;
        case 'previous_grade':
            if (value && (parseFloat(value) < 0 || parseFloat(value) > 100)) {
                isValid = false;
            }
            break;
    }
    
    // Add error styling if invalid
    if (!isValid) {
        field.classList.add('error');
    }
    
    return isValid;
}

// Add CSS for error styling
const style = document.createElement('style');
style.textContent = `
    .error {
        border-color: #dc3545 !important;
        box-shadow: 0 0 0 0.2rem rgba(220, 53, 69, 0.25) !important;
    }
`;
document.head.appendChild(style);