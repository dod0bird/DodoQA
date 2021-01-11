$(document).ready(function() {
    $('#question-button').click(function() {
        var question = $('#question-field').val();
        console.log(question);

        $.ajax({
            type: 'POST',
            url: '/api/ask',
            data: JSON.stringify({question: question}),
            contentType: 'application/json',
            dataType: 'json'
        }).done(function(response) {
            console.log('response: ' + response);
        }).fail(function(jqXHR, textStatus) {
            console.log("Request failed: " + textStatus);
        });
    });
});
