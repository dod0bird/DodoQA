$(document).ready(function() {
    $('#question-button').click(function() {
        var question = $('#question-field').val();
        console.log(question);

        $('#answers').html("<div class='spinner-border text-info' role='status'></div>");

        $.ajax({
            type: 'POST',
            url: '/api/ask',
            data: JSON.stringify({question: question}),
            contentType: 'application/json',
            dataType: 'json'
        }).done(function(response) {
            console.log('response: ' + response);
            $('#answers').empty();
            if (response.length === 0) {
                $('#answers').html("<div class='alert alert-secondary' role='alert'>No answers found.</div>");
            } else {
                response.forEach(function(answer) {
                    var index = answer.context.indexOf(answer.answer);
                    var highlightedContext = answer.context;
                    if (index >= 0) {
                        var highlightedContext = answer.context.substring(0, index) + "<span class='bg-warning'>" + answer.context.substring(index, index+answer.answer.length) + "</span>" + answer.context.substring(index + answer.answer.length);
                    }
                    $('#answers').append(
                        $(
                            "<div class='card mb-3 border-info'><div class='card-body'>" +
                            "<blockquote class='blockquote'><p class=''>" + highlightedContext + "</p>" +
                            "<footer class='blockquote-footer'><cite title='Source Title'>" +
                            "<a href='" + answer.url + "' target='_blank'>" + answer.url + "</a></cite></footer>" +
                            "</blockquote>" +
                            "<p><b>Search score:&nbsp;</b>" + answer.retrieval_score + "&emsp;<b>Reader score:&nbsp;</b>" + answer.reader_score +
                            "&emsp;<b>Combined score:&nbsp;</b>" + answer.combined_score + "</p>" +
                            "</div></div>"
                        )
                    );
                });
            }
        }).fail(function(jqXHR, textStatus) {
            $('#answers').html("<div class='alert alert-warning' role='alert'>An error occurred.</div>");
            console.log("Request failed: " + textStatus);
        });
    });
});
