<?php
defined('MOODLE_INTERNAL') || die();

/**
 * Adds LLM Chat page to navigation.
 *
 * @param global_navigation $navigation
 * @return void
 */
function local_chatbot_extend_navigation(global_navigation $navigation): void {
    if (!isloggedin() || isguestuser()) {
        return;
    }

    $context = context_system::instance();
    if (!has_capability('local/chatbot:view', $context)) {
        return;
    }

    $url = new moodle_url('/local/chatbot/index.php');
    $node = navigation_node::create(
        get_string('pluginname', 'local_chatbot'),
        $url,
        navigation_node::TYPE_CUSTOM,
        null,
        'local_chatbot'
    );
    $navigation->add_node($node);
}
