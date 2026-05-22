<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Moodle is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Moodle.  If not, see <http://www.gnu.org/licenses/>.

defined('MOODLE_INTERNAL') || die();

$string['pluginname'] = 'LLM Chat';
$string['settingsheading'] = 'LLM Chat settings';
$string['projectpath'] = 'Project path';
$string['projectpath_desc'] = 'Absolute path to the LLM project folder.';
$string['pythonpath'] = 'Python executable path';
$string['pythonpath_desc'] = 'Absolute path to python executable used by the plugin.';
$string['runnerfile'] = 'Runner file name';
$string['runnerfile_desc'] = 'Backend runner filename inside the project path.';

$string['chatbot:view'] = 'View LLM Chat page.';

$string['materialstitle'] = 'Materials';
$string['uploaddesc'] = 'Select class/topic or upload manual PDF/TXT materials to define the active RAG corpus.';
$string['manualuploadtitle'] = 'Manual upload';
$string['manualuploadlabel'] = 'Upload PDF/TXT materials';
$string['manualuploadbutton'] = 'Upload materials';
$string['manualclearbutton'] = 'Clear uploaded materials';
$string['manualuploadhelp'] = 'Manual upload becomes the active corpus and temporarily disables topic selection.';
$string['manualuploadreadonly'] = 'Upload is visible here, but only teacher-like users can change the active corpus.';
$string['manualuploadrequired'] = 'Choose at least one PDF or TXT file first.';
$string['manualuploading'] = 'Uploading materials...';
$string['manualuploadsuccess'] = 'Manual materials uploaded. Topic selection is now disabled.';
$string['manualcleared'] = 'Manual uploaded materials cleared. Topic selection is enabled again.';
$string['manualmodeactive'] = 'Manual upload is active. Clear uploaded materials to use class/topic materials again.';
$string['topicmaterialstitle'] = 'Class/topic materials';
$string['classlabel'] = 'Class';
$string['classplaceholder'] = 'Select class';
$string['topiclabel'] = 'Topic';
$string['topicplaceholder'] = 'Select topic';
$string['executiontitle'] = 'Execution';
$string['evaluationmodetitle'] = 'Evaluation mode';
$string['modellabel'] = 'Modes';
$string['modeplaceholder'] = 'Select mode';
$string['mode_llm_only'] = 'LLM only';
$string['mode_rag_ollama'] = 'RAG Ollama';
$string['mode_rag_bert'] = 'RAG BERT';
$string['mode_rag_msmarco'] = 'RAG MSMARCO';
$string['embeddingconfigtitle'] = 'Embedding configuration';
$string['embeddingconfigactive'] = 'Active embedding';
$string['embeddingconfigbackend'] = 'Default backend';
$string['embeddingconfigollama'] = 'Ollama embedding model';
$string['embeddingconfigbert'] = 'BERT embedding model';
$string['embeddingconfigllmonly'] = 'No embedding is used in LLM-only mode.';
$string['evallabel'] = 'Enable evaluation mode';
$string['evalsourcelabel'] = 'Evaluation source';
$string['evalsourcechat'] = 'Direct chat';
$string['evalsourcedataset'] = 'Answer-run dataset';
$string['evalquestionidlabel'] = 'Question ID';
$string['evalrunidlabel'] = 'Run ID';
$string['evaldatasettitle'] = 'Answer-run dataset';
$string['evaldatasetlabel'] = 'Upload dataset JSON';
$string['evaldatasetrunslabel'] = 'Runs per question';
$string['evaldatasetrunbutton'] = 'Run answer-run dataset';
$string['evaldatasetrunning'] = 'Running answer-run dataset...';
$string['evaldatasetsuccess'] = 'Answer-run dataset finished.';
$string['evalformtitle'] = 'Answer evaluation';
$string['evalformscalehelp'] = 'Rate each item from 1 (strongly disagree) to 5 (strongly agree).';
$string['evalformcorrectness'] = 'The chatbot answer is correct according to the learning material.';
$string['evalformgroundedness'] = 'The chatbot answer is supported by relevant material or context.';
$string['evalformrelevance'] = 'The chatbot answer matches the question being asked.';
$string['evalforminstructioncompliance'] = 'The chatbot answer follows the instruction given in the question.';
$string['evalformneedalignment'] = 'The chatbot answer helps my learning need at this moment.';
$string['evalformscaffoldingquality'] = 'The chatbot answer helps me understand the material step by step.';
$string['evalformclarity'] = 'The chatbot answer is easy to understand.';
$string['evalformcommentlabel'] = 'Optional comment';
$string['evalformcommentplaceholder'] = 'What was helpful or still needs improvement?';
$string['evalformsubmit'] = 'Submit evaluation';
$string['evalformsubmitted'] = 'Evaluation saved for this answer.';
$string['evalformscoreplaceholder'] = 'Choose score';
$string['evalformsaving'] = 'Saving evaluation...';
$string['evalformsaveerror'] = 'Failed to save evaluation.';
$string['evalformrequired'] = 'Please choose a score for every evaluation item before submitting.';
$string['evalformsaveok'] = 'Evaluation saved successfully.';
$string['uploadedtitle'] = 'Materials';
$string['nofiles'] = 'No active materials loaded.';
$string['nocoursesavailable'] = 'No enrolled course found';
$string['statusready'] = 'RAG ready';
$string['statusnodocs'] = 'No materials selected';
$string['chatheader'] = 'Chat with class materials';
$string['chatplaceholder'] = 'Ask a question about selected class materials...';
$string['sendbutton'] = 'Send';
$string['chatusagelabel'] = 'Usage';
$string['clearhistorylabel'] = 'Clear history';
$string['clearhistoryconfirm'] = 'Clear this chat history?';
$string['defaultgreeting'] = 'Hello. Select class and topic in the left panel, then ask about the materials.';
$string['thinking'] = 'Thinking...';
$string['chaterror'] = 'Failed to process chat request.';
$string['previewtitle'] = 'Preview';
$string['previewempty'] = 'No preview available.';
$string['previewloading'] = 'Generating preview...';
$string['previewerror'] = 'Failed to generate preview.';
$string['previewopenpdf'] = 'Open PDF in new tab';
$string['previewpdffallback'] = 'If the PDF does not render here, open it in a new tab.';
$string['refreshembeddingbutton'] = 'Refresh embedding';
$string['refreshembeddingloading'] = 'Refreshing embedding index...';
$string['refreshembeddingrequired'] = 'Select a document first.';
$string['refreshembeddingok'] = 'Embedding index refreshed for the active corpus.';
$string['refreshembeddingerror'] = 'Failed to refresh embedding index.';
