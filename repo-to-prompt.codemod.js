
/**
 * @param {vscode} vscode the entry to vscode plugin api
 * @param {vscode.Uri} selectedFile currently selected file in vscode explorer
 * @param {vscode.Uri[]} selectedFiles currently multi-selected files in vscode explorer
 */
async function run(vscode, selectedFile, selectedFiles) {
    console.log('You can debug the script with console.log')
    // remove useless file from selectedFiles
    selectedFiles = selectedFiles.filter(file => !file.path.endsWith('.env') && !file.path.endsWith('.lock') && !file.path.endsWith('LICENSE'));
    const lines = [];
    for (const file of selectedFiles) {
        lines.push('<file path="' + file.path + '">')
        lines.push(' ')
        lines.push('```')
        lines.push(new TextDecoder().decode(await vscode.workspace.fs.readFile(file)))
        lines.push('```')
        lines.push('</file>')
    }
    await vscode.env.clipboard.writeText(lines.join('\n'))
    vscode.window.showInformationMessage('Copied to clipboard as Prompt XML.')
}
await run(vscode, selectedFile, selectedFiles);
