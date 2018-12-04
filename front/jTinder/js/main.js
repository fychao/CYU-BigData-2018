/**
 * jTinder initialization
 */
$("#tinderslide").jTinder({
	// dislike callback
    onDislike: function (item) {
	    // set the status text
	    $("#dislike_str").html($("#dislike_str").html() + $("#questions-"+(item.index()+1)).html());
        //$('#status').html($('#status').html() + (item.index()+1));

        if( item.index() == 0) $("#submit_result").show();
    },
	// like callback
    onLike: function (item) {
	    // set the status text
        //$('#status').html($('#status').html() + (item.index()+1));
        $("#like_str").html($("#like_str").html() + $("#questions-"+(item.index()+1)).html());
        if( item.index() == 0) $("#submit_result").show();
    },
	animationRevertSpeed: 200,
	animationSpeed: 400,
	threshold: 1,
	likeSelector: '.like',
	dislikeSelector: '.dislike'
});

/**
 * Set button action to trigger jTinder like & dislike.
 */
$('.actions .like, .actions .dislike').click(function(e){
	e.preventDefault();
	$("#tinderslide").jTinder($(this).attr('class'));
});