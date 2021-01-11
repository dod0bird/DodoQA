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
            $('#answers').empty();
            response.forEach(function(answer) {
                $('#answers').append($("<div class='card mb-3 border-info'><div class='card-body'>" + answer + "</div></div>"));
            });
        }).fail(function(jqXHR, textStatus) {
            console.log("Request failed: " + textStatus);
        });
    });
});
