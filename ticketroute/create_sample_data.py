"""Synthetic tickets: unique scenarios split before training."""
import csv
from pathlib import Path
EXAMPLES = {
'billing problem': [
('I was charged twice for the same subscription.', 'train'),
('The invoice shows a higher amount than the plan price.', 'train'),
('My card was billed after I cancelled the subscription.', 'train'),
('A payment succeeded but the invoice still says unpaid.', 'train'),
('Please explain the tax amount on my latest bill.', 'train'),
('The renewal fee appeared two times on my statement.', 'train'),
('I need a corrected receipt for last month\'s purchase.', 'train'),
('My bank shows a debit although checkout reported payment failure.', 'train'),
('My discount was not applied to the latest invoice.', 'validation'),
('Why was I billed when I am still on the free trial?', 'validation'),
('The refunded amount on my statement is lower than expected.', 'test'),
('My annual plan payment was taken, but I received no receipt.', 'test')],
'account access problem': [
('My password reset link has expired before I could use it.', 'train'),
('I cannot log in after changing my phone number.', 'train'),
('The sign-in verification code never arrives by email.', 'train'),
('My account is locked after several incorrect passwords.', 'train'),
('I lost access to the authenticator for my profile.', 'train'),
('The site says my email is not registered when I sign in.', 'train'),
('I cannot accept the team invitation with my existing account.', 'train'),
('My password works but the login page returns me to sign in.', 'train'),
('The reset password email does not reach my mailbox.', 'validation'),
('How can I regain access after my account was disabled?', 'validation'),
('My two-factor security code is rejected during login.', 'test'),
('After changing my email address I cannot access my profile.', 'test')],
'technical error': [
('The dashboard displays a blank page in my browser.', 'train'),
('The application crashes whenever I upload a PDF.', 'train'),
('A server error appears when I export my report.', 'train'),
('The mobile app freezes on its loading screen.', 'train'),
('The search button does nothing after I press it.', 'train'),
('My uploaded image appears broken on the profile page.', 'train'),
('The notification settings page shows an unexpected error.', 'train'),
('CSV downloads are corrupted when opened in Excel.', 'train'),
('The reports page keeps timing out after loading.', 'validation'),
('I receive a 500 error when saving my settings.', 'validation'),
('The dashboard charts disappear after the page refreshes.', 'test'),
('My uploaded spreadsheet fails to process with a server error.', 'test')],
'feature request': [
('Please add a dark mode to the dashboard.', 'train'),
('Could you support exporting reports as PDF?', 'train'),
('I would like a way to schedule weekly summary emails.', 'train'),
('Can you add a filter by customer region?', 'train'),
('It would be useful to sort results by creation date.', 'train'),
('Please let us customize dashboard colors.', 'train'),
('Could the mobile app support offline viewing?', 'train'),
('Add an option to pin favorite reports.', 'train'),
('Please add keyboard shortcuts for common actions.', 'validation'),
('I would like to create custom report templates.', 'validation'),
('Can the app include a calendar view for team tasks?', 'test'),
('Please introduce a bulk edit option for selected records.', 'test')],
}
OUT_OF_SCOPE = [
'What time is it in Hyderabad?', 'Please tell me a joke about a robot.',
'What is the weather forecast for tomorrow?', 'Explain how to cook pasta.',
'My package arrived damaged and I need an exchange.',
'The delivery driver left my parcel at the wrong address.',
'Can I speak to a human customer service representative?',
'What are your office opening hours?']


def write_files():
    root = Path(__file__).resolve().parent / 'data'
    root.mkdir(exist_ok=True)
    rows = []
    for label, samples in EXAMPLES.items():
        for index, (text, split) in enumerate(samples, 1):
            group = f'{label.replace(" ", "_")}_{index:02}'
            rows.append({'text': text, 'label': label,
                         'group_id': group, 'split': split})
    for split in ('train', 'validation', 'test'):
        with (root / f'{split}.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'label', 'group_id'])
            writer.writeheader()
            writer.writerows({k: row[k] for k in ('text', 'label', 'group_id')}
                             for row in rows if row['split'] == split)
    with (root / 'out_of_scope.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'expected'])
        writer.writeheader()
        writer.writerows({'text': text, 'expected': 'manual review'} for text in OUT_OF_SCOPE)
    print('Created 32 train, 8 validation, 8 test and 8 out-of-scope rows.')

if __name__ == '__main__': write_files()
