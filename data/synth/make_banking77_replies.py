"""Synthetic reply variants for Banking77 intents (authored by Claude Code, 2026-11-21).

Real customer queries (Banking77) are paired with one of these generic, safe replies for their
intent. The replies contain no account data and no specific fees/limits (those vary by bank).
Run:  python data/synth/make_banking77_replies.py  -> banking77_replies.jsonl + banking77_labels.json
"""
import json
from pathlib import Path

R = {
"Refund_not_showing_up": [
 "Sorry your refund isn't showing yet. Merchant refunds can take several working days to reach your account after the merchant processes them.",
 "Refunds appear once the merchant sends them, which can take a few days. If it's been longer than the merchant said, contact us with the refund details.",
 "Please check with the merchant that the refund was issued. If they confirm it was sent and it still isn't showing, let us know and we'll trace it."],
"activate_my_card": [
 "You can activate your card in the app: open Cards, select the new card and tap Activate. You'll be asked to confirm a few details.",
 "To activate your card, go to the Cards section of the app, choose the card and follow the Activate steps. It's ready to use right after.",
 "Activation only takes a minute in the app. Open Cards, pick your new card and tap Activate. Contact us if the option doesn't appear."],
"age_limit": [
 "You need to be at least the minimum legal age in your country to open an account. You can check the exact requirement in the app's Help section.",
 "Account eligibility depends on your age and country of residence. The Help section in the app lists the minimum age that applies to you.",
 "There is a minimum age to open an account, which varies by country. Please check the eligibility page in Help for your country's requirement."],
"apple_pay_or_google_pay": [
 "Yes, you can add your card to Apple Pay or Google Pay. Open the card in the app and tap Add to Wallet, then follow the prompts.",
 "To use Apple Pay or Google Pay, open Cards in the app, choose your card and select Add to Wallet. Your phone will guide you through the rest.",
 "You can link your card to Apple Pay or Google Pay straight from the app. If the button doesn't show, check that your app and phone are up to date."],
"atm_support": [
 "You can use your card at any ATM that accepts its card network. Some ATM operators charge their own fee, which the machine will show before you confirm.",
 "Your card works at ATMs displaying your card network's logo. Check the app for any withdrawal allowances that apply to your plan.",
 "Most ATMs that accept your card network will work. If a withdrawal fails, check your PIN and daily limit in the app, then try another ATM."],
"automatic_top_up": [
 "You can turn on automatic top-up in the app under Top up > Auto top-up. Choose the threshold and amount, and we'll top up when your balance drops below it.",
 "Auto top-up keeps your balance from running low. Set it up in the Top up section by choosing a trigger balance and a top-up amount.",
 "To change or disable automatic top-up, go to Top up > Auto top-up in the app and update the settings or switch it off."],
"balance_not_updated_after_bank_transfer": [
 "Bank transfers can take a little time to arrive depending on the sending bank and payment method. If it's been longer than expected, please share the transfer reference so we can trace it.",
 "Sorry your balance hasn't updated yet. Incoming transfers usually arrive within the stated processing time; if not, send us the payment details from your bank and we'll investigate.",
 "Transfers from another bank can be delayed by processing times or checks. If the money still isn't showing, contact us with the sender's confirmation and we'll look into it."],
"balance_not_updated_after_cheque_or_cash_deposit": [
 "Cheque and cash deposits can take a few working days to clear. If it's been longer than that, please send us the deposit receipt so we can check.",
 "Sorry for the wait. Deposits need to be processed before they appear in your balance. If it's overdue, share the deposit details and we'll investigate.",
 "Deposited funds appear once they've cleared, which can take some time. Contact us with the receipt if your balance still hasn't updated."],
"beneficiary_not_allowed": [
 "Some beneficiaries can't be added for security or regulatory reasons. Please double-check the details, and contact us if you need help understanding why it was blocked.",
 "Sorry you couldn't add that beneficiary. This can happen if details don't match or the recipient isn't supported. Our team can review it with you.",
 "Adding a beneficiary may fail if the account details are incorrect or the destination isn't supported. Check the details and reach out to us if it keeps failing."],
"cancel_transfer": [
 "If the transfer is still pending, you may be able to cancel it in the app from the transaction details. Once completed, it can't be cancelled, but we can try to help recover it.",
 "Open the transfer in your activity list; if it hasn't been processed yet you'll see a Cancel option. If not, contact us as soon as possible.",
 "Transfers can only be cancelled before they're processed. Check the transaction in the app, and get in touch with us right away if there's no cancel option."],
"card_about_to_expire": [
 "We'll send you a replacement card before your current one expires. You can also check the status or order a new card in the Cards section of the app.",
 "No need to worry: a new card is usually issued automatically before expiry. Make sure your address is up to date in the app.",
 "If your card is expiring soon, you can request a replacement from the Cards section. Please confirm your delivery address first."],
"card_acceptance": [
 "Your card can be used wherever its card network is accepted, in-store and online. A few merchants may not accept certain card types.",
 "Most shops and websites that accept your card network will accept your card. If a payment fails, check your card settings and limits in the app.",
 "You can pay anywhere your card network is accepted. If a specific merchant declines it, let us know and we can check what happened."],
"card_arrival": [
 "Sorry your card hasn't arrived yet. You can track its delivery status in the Cards section of the app. If it's overdue, we can cancel it and send a new one.",
 "Card delivery times vary by location. Check the delivery status in the app, and contact us if it has taken longer than the estimated time.",
 "If your card still hasn't arrived, please confirm your address in the app and contact us. We can freeze the old card and order a replacement."],
"card_delivery_estimate": [
 "Delivery time depends on your location and the shipping option. You'll see an estimated delivery date in the Cards section once the card is ordered.",
 "Cards usually arrive within the delivery window shown in the app when you order. Express options may be available in some countries.",
 "You can find the estimated delivery date for your card in the app under Cards. We'll notify you when it's on its way."],
"card_linking": [
 "To link a card you already have, open Cards in the app, choose Link card, and enter the details shown on the card.",
 "You can link your card in the app by going to Cards > Link a card and following the steps. Make sure the card is in your name.",
 "Linking takes just a moment in the Cards section. If the app doesn't accept the card, check the details and contact us."],
"card_not_working": [
 "Sorry your card isn't working. Please check in the app that the card isn't frozen and that the payment type is enabled. If the problem continues, we can issue a replacement.",
 "First, make sure your card isn't frozen or expired and that you have enough balance. If it still doesn't work, contact us and we'll investigate.",
 "Card issues can be caused by security settings, limits, or a damaged chip. Check your card settings in the app and reach out if it keeps failing."],
"card_payment_fee_charged": [
 "Some card payments can carry a fee depending on the currency, merchant or your plan. Open the transaction in the app to see the fee breakdown.",
 "Sorry for the surprise fee. You can see why it was charged in the transaction details. Contact us if you think it was charged incorrectly.",
 "Fees may apply to certain card payments, such as those in another currency. Please check your plan's fee details in the app."],
"card_payment_not_recognised": [
 "If you don't recognise a card payment, please freeze your card in the app right away and contact us. We'll help you investigate and dispute it if needed.",
 "Sorry to hear that. Check the merchant name in the transaction details first, as it can differ from the shop's name. If it's still unfamiliar, freeze your card and report it.",
 "For your security, freeze your card in the app and report the payment from the transaction screen. Our team will review it."],
"card_payment_wrong_exchange_rate": [
 "Exchange rates can change between payment and settlement, and some merchants apply their own conversion. Check the transaction details, and contact us if the rate looks wrong.",
 "Sorry about the confusion. If you paid in the merchant's currency, our rate applies; if the merchant converted it, their rate applies. We can review the payment with you.",
 "The rate shown may differ due to timing or merchant-side conversion. Share the transaction details with us and we'll check it."],
"card_swallowed": [
 "Sorry the ATM kept your card. Please freeze it in the app right away and order a replacement from the Cards section.",
 "If an ATM swallowed your card, freeze or cancel it in the app for safety, then request a new card. You can also contact the ATM's bank.",
 "For your security, freeze the card in the app immediately. You can order a replacement in the Cards section."],
"cash_withdrawal_charge": [
 "Cash withdrawals may be free up to an allowance and charged after that, depending on your plan. Some ATM operators also add their own fee.",
 "You can see any withdrawal fee in the transaction details. Check your plan's allowance in the app to avoid unexpected charges.",
 "Sorry about the charge. Fees can come from your plan's limits or the ATM operator. Contact us if you think it was applied in error."],
"cash_withdrawal_not_recognised": [
 "If you don't recognise a cash withdrawal, freeze your card in the app immediately and contact us so we can investigate.",
 "Sorry to hear that. Please freeze your card and report the withdrawal from the transaction screen. We'll look into it urgently.",
 "For your safety, freeze the card right away and get in touch with us. We'll review the withdrawal and next steps."],
"change_pin": [
 "You can change your PIN at most ATMs that support PIN changes, using your current PIN. Some apps also let you view your PIN in the card settings.",
 "To change your PIN, use an ATM with a PIN-management option. You'll need your current PIN to set a new one.",
 "PIN changes can be made at compatible ATMs. If you've forgotten your PIN, you can view it securely in the app's card settings."],
"compromised_card": [
 "If you think your card is compromised, freeze it in the app immediately and order a replacement. Contact us to report any suspicious transactions.",
 "Please freeze your card right away from the Cards section, then cancel it and request a new one. We'll help you review recent payments.",
 "For your security, freeze and cancel the card in the app, then order a replacement. Report any payments you don't recognise."],
"contactless_not_working": [
 "Contactless can be disabled until you make a chip-and-PIN payment first. Try inserting your card once, then contactless should work.",
 "Check in the app that contactless payments are enabled for your card. Also note there may be a per-payment contactless limit.",
 "If contactless isn't working, make a chip-and-PIN payment and check your card settings in the app. Contact us if it still fails."],
"country_support": [
 "Our service is available in a set of supported countries. You can check the full list in the Help section of the app.",
 "Availability depends on your country of residence. Please see the supported countries list in Help.",
 "Whether you can open an account depends on where you live. The Help section lists all the countries we currently support."],
"declined_card_payment": [
 "Sorry your payment was declined. Check in the app that your card isn't frozen, you have enough balance, and the payment type is allowed.",
 "Declines can happen due to limits, security checks or merchant issues. The transaction details in the app may show the reason.",
 "Please check your card status and balance in the app. If everything looks fine and payments keep being declined, contact us."],
"declined_cash_withdrawal": [
 "Sorry your withdrawal was declined. Please check your balance, daily withdrawal limit, and that you entered the correct PIN.",
 "ATM withdrawals can be declined because of limits, PIN errors or ATM issues. Try another ATM, and contact us if it keeps happening.",
 "Check in the app that your card isn't frozen and your withdrawal limit hasn't been reached. We can help if the problem continues."],
"declined_transfer": [
 "Sorry your transfer was declined. Please check the recipient details and your balance. Some transfers are also declined for security reasons.",
 "A transfer can be declined if details are wrong, limits are reached or checks are needed. The app may show the reason in the transaction details.",
 "Please double-check the recipient's details and try again. If it's declined again, contact us and we'll review it."],
"direct_debit_payment_not_recognised": [
 "If you don't recognise a direct debit, check the merchant name in the transaction details. You can cancel the mandate in the app and contact us to dispute it.",
 "Sorry to hear that. You can view and cancel direct debits in the app. Please report the payment so we can help you investigate.",
 "Please review the direct debit details and cancel it if it isn't yours. Contact us to raise a dispute."],
"disposable_card_limits": [
 "Disposable virtual cards are designed for one-time use and may have limits on how many you can create. Check your plan details in the app.",
 "Limits on disposable cards depend on your plan. You can see how many you can generate in the Cards section.",
 "Each disposable card is replaced after one payment. Your plan determines how many you can create; see the app for details."],
"edit_personal_details": [
 "You can update most personal details in the app under Profile. Some changes, like your legal name, may require verification.",
 "To change your details, go to Profile in the app and edit the relevant field. We may ask for a document for certain changes.",
 "Personal details can be edited from your Profile. Contact us if a field can't be changed in the app."],
"exchange_charge": [
 "Currency exchange may be free up to an allowance and carry a fee beyond that, depending on your plan and the time of day. Check the fee details in the app.",
 "Exchange fees depend on your plan, the currency and market hours. The app shows any fee before you confirm the exchange.",
 "Sorry about the unexpected charge. You can see exchange fee details in the app; contact us if you think it was applied wrongly."],
"exchange_rate": [
 "You can see the current exchange rate in the app before making an exchange or payment. Rates move with the market.",
 "Our exchange rates are based on market rates and are shown in the app at the time of exchange.",
 "Open the Exchange section in the app to see the live rate for your currency pair before you confirm."],
"exchange_via_app": [
 "You can exchange currencies in the app: open Exchange, choose the currencies and amount, and confirm.",
 "To exchange money, go to the Exchange section, select the currencies, enter the amount and tap Exchange.",
 "Currency exchange is available directly in the app under Exchange. The rate and any fee are shown before you confirm."],
"extra_charge_on_statement": [
 "Sorry about the extra charge. Open the transaction in the app to see what it relates to. Contact us if it still doesn't look right.",
 "Some charges come from fees, currency conversion or the merchant. Check the transaction details, and we can investigate if needed.",
 "If you see a charge you don't expect, review its details in the app and get in touch with us so we can look into it."],
"failed_transfer": [
 "Sorry your transfer failed. Please check the recipient details and your balance, then try again. Any deducted amount is usually returned.",
 "Transfers can fail due to incorrect details or checks by the receiving bank. The app may show the reason; contact us if you need help.",
 "If a transfer failed, the money normally comes back to your account. Double-check the details and try again, or contact us."],
"fiat_currency_support": [
 "We support a range of currencies for holding and exchange. You can see the full list in the app's Exchange section.",
 "The currencies you can hold and exchange are listed in the app. Support may vary by country.",
 "Please check the Exchange section in the app for the list of supported currencies."],
"get_disposable_virtual_card": [
 "You can create a disposable virtual card in the app under Cards > Add card > Disposable. It's replaced automatically after each payment.",
 "To get a disposable card, open Cards, tap Add and choose Disposable virtual card. Availability may depend on your plan.",
 "Disposable virtual cards can be generated in the Cards section. They're great for one-off online payments."],
"get_physical_card": [
 "You can order a physical card in the app under Cards > Order physical card. Make sure your delivery address is correct.",
 "To get a physical card, open the Cards section and choose Order a physical card. Delivery details are shown during ordering.",
 "Ordering a physical card takes a few taps in the Cards section. We'll notify you when it ships."],
"getting_spare_card": [
 "You may be able to order an additional card in the Cards section of the app, depending on your plan.",
 "To get a spare card, open Cards and tap Order new card. Any fee for extra cards is shown before you confirm.",
 "Additional cards can be ordered from the app. Check your plan for how many cards you can have."],
"getting_virtual_card": [
 "You can create a virtual card in the app under Cards > Add card > Virtual. It's ready to use online immediately.",
 "Virtual cards can be added instantly in the Cards section. You can also add them to Apple Pay or Google Pay.",
 "To get a virtual card, open Cards, tap Add and choose Virtual card. Availability may depend on your plan."],
"lost_or_stolen_card": [
 "Sorry to hear that. Please freeze or cancel your card in the app straight away, then order a replacement from the Cards section.",
 "For your security, freeze your card immediately in the app. If you can't find it, cancel it and request a new one.",
 "Please freeze the card in the app right now and report any payments you don't recognise. You can order a new card afterwards."],
"lost_or_stolen_phone": [
 "Sorry to hear that. Please contact us as soon as possible so we can secure your account. You can also log in from another device to freeze your cards.",
 "If your phone is lost or stolen, get in touch with us right away to block access. Change your passwords and freeze your cards from another device if you can.",
 "For your security, contact us immediately so we can lock your account on the lost device. You can then log in again on a new phone."],
"order_physical_card": [
 "You can order a physical card from the Cards section of the app. Confirm your address and choose a delivery option.",
 "To order a card, open Cards and tap Order physical card. You'll see the delivery estimate before confirming.",
 "Physical cards can be ordered in a few steps in the app. We'll let you know when it's on its way."],
"passcode_forgotten": [
 "No problem. On the login screen, tap Forgot passcode and follow the steps to verify your identity and set a new one.",
 "You can reset your passcode from the app's login screen using the Forgot passcode option.",
 "To reset a forgotten passcode, tap Forgot passcode when logging in. You may be asked to verify your identity."],
"pending_card_payment": [
 "Card payments show as pending until the merchant completes them, which can take a few days. If it's cancelled, the funds are released automatically.",
 "A pending payment means the merchant hasn't finalised it yet. It will either complete or drop off within the usual processing time.",
 "Pending card payments are normal and usually complete within a few days. Contact the merchant if it stays pending for a long time."],
"pending_cash_withdrawal": [
 "Cash withdrawals can show as pending until the ATM operator confirms them. This usually resolves within a few days.",
 "A pending withdrawal means it hasn't settled yet. If you didn't receive the cash, contact us and we'll investigate.",
 "Pending withdrawals normally settle automatically. Let us know if it stays pending or you didn't get your money."],
"pending_top_up": [
 "Top-ups can stay pending while they're being processed or checked. This usually resolves quickly.",
 "Sorry for the wait. A pending top-up should complete shortly. If it doesn't, contact us with the top-up details.",
 "Your top-up is being processed. If it stays pending longer than expected, please get in touch."],
"pending_transfer": [
 "Transfers can stay pending while they're processed or checked. Most complete within the stated processing time.",
 "A pending transfer is still being processed. You can see its status in the app; contact us if it's taking too long.",
 "Sorry for the delay. Some transfers need extra checks. If it's still pending after the expected time, please reach out."],
"pin_blocked": [
 "Your PIN can be blocked after too many incorrect attempts. You can usually unblock it in the app's card settings or at an ATM.",
 "Sorry about that. Check the card settings in the app for an Unblock PIN option. Contact us if you can't find it.",
 "A PIN is blocked after several wrong entries. You can view your PIN in the app and unblock it from the card settings."],
"receiving_money": [
 "You can receive money using your account details, found in the app under Account details. Share them with the sender.",
 "To receive money, give the sender your account details from the app. Incoming payments appear once processed.",
 "Your account details for receiving money are in the app. Transfer times depend on the sender's bank and payment method."],
"request_refund": [
 "Refunds are usually issued by the merchant. Please contact them first; if they don't help, contact us and we can look into a dispute.",
 "To get a refund, ask the merchant to process it back to your card. Refunds can take a few days to appear.",
 "Sorry for the trouble. Merchant refunds come back to your card once they're processed. If the merchant won't help, we can review a dispute."],
"reverted_card_payment?": [
 "A reverted payment means the merchant cancelled or didn't complete it, so the funds go back to your balance.",
 "If a card payment was reverted, the money is returned to your account. Contact the merchant if you expected the payment to go through.",
 "Reverted payments happen when a merchant cancels the authorisation. The amount should be back in your balance shortly."],
"supported_cards_and_currencies": [
 "You can top up with most major debit and credit cards, and hold and exchange a range of currencies. See the app for the full list.",
 "Supported cards and currencies are listed in the app's Help section. Support can vary by country.",
 "Please check the Help section for which cards you can top up with and which currencies are supported."],
"terminate_account": [
 "We're sorry to see you go. You can close your account from the app's Profile settings. Please withdraw any remaining balance first.",
 "To close your account, go to Profile > Close account in the app and follow the steps. Make sure your balance is empty.",
 "You can request account closure in the app. Withdraw or transfer your remaining funds before closing."],
"top_up_by_bank_transfer_charge": [
 "Topping up by bank transfer is usually free on our side, but your bank may charge a fee. Check the app's fee details for your plan.",
 "We generally don't charge for bank transfer top-ups, though the sending bank might. See the fees section in the app.",
 "Bank transfer top-ups are typically free from us. If you were charged, contact us and we'll check where it came from."],
"top_up_by_card_charge": [
 "Card top-ups may carry a fee depending on the card type and where it was issued. The app shows any fee before you confirm.",
 "Some card top-ups are charged, for example with certain cards. You'll see the fee during the top-up.",
 "Sorry about the charge. Fees for card top-ups depend on the card used; check the details in the app."],
"top_up_by_cash_or_cheque": [
 "Cash or cheque top-ups may not be supported in all regions. Please check the Top up section of the app for available options.",
 "Available top-up methods are listed in the app. Cash and cheque options depend on your country.",
 "You can see whether cash or cheque top-ups are supported in your country in the app's Top up section."],
"top_up_failed": [
 "Sorry your top-up failed. Please check your card details and that your bank allows the payment, then try again.",
 "Top-ups can fail due to card issues or bank checks. Try again or use a different method, and contact us if it continues.",
 "If a top-up failed, any held amount should be released by your bank. Double-check the details and try again."],
"top_up_limits": [
 "Top-up limits depend on your plan and verification level. You can see your limits in the app's Top up section.",
 "There may be limits on how much and how often you can top up. Check the app for your current limits.",
 "Your top-up limits are shown in the app. Verifying your identity may increase them."],
"top_up_reverted": [
 "A reverted top-up means the payment didn't complete and the money is returned to your card. This can happen after a bank check.",
 "Sorry about that. If your top-up was reverted, the funds go back to the source card. You can try again or use another method.",
 "Reverted top-ups are returned to the card used. Contact your card issuer or us if you need more help."],
"topping_up_by_card": [
 "You can top up with a card in the app: go to Top up, choose Card, enter the amount and confirm.",
 "To top up by card, open the Top up section and select your saved card or add a new one.",
 "Card top-ups are usually instant. Open Top up in the app, choose a card and enter the amount."],
"transaction_charged_twice": [
 "Sorry about that. Sometimes a duplicate is just a pending authorisation that drops off. If both payments complete, contact us and we'll help.",
 "If you were charged twice, check whether one of the payments is still pending. If both settle, please report it so we can investigate.",
 "Duplicate charges often resolve automatically. If they don't, contact the merchant or let us know and we'll review it."],
"transfer_fee_charged": [
 "Some transfers carry a fee depending on the destination, currency or your plan. The fee is shown in the transaction details.",
 "Sorry for the unexpected fee. Check the transfer details in the app to see why it applied, and contact us if it looks wrong.",
 "Transfer fees depend on the type of transfer. You can see the breakdown in the app."],
"transfer_into_account": [
 "To transfer money into your account, use your account details from the app with your other bank.",
 "You can add money by bank transfer using the account details shown in the app.",
 "Share your account details from the app with the sending bank to move money into your account."],
"transfer_not_received_by_recipient": [
 "Sorry the recipient hasn't received it yet. Transfers can take time depending on the destination. Please check the status in the app and share the details with us if it's overdue.",
 "If the transfer shows as completed, ask the recipient to check with their bank. We can provide a payment confirmation to help trace it.",
 "Check that the recipient's details were correct. If the transfer is late, contact us with the transfer details and we'll investigate."],
"transfer_timing": [
 "Transfer times depend on the destination and payment method. The app shows an estimate when you send.",
 "Most transfers arrive within the estimated time shown in the app. International transfers can take longer.",
 "You'll see the expected arrival time when you make the transfer. Some transfers need extra checks."],
"unable_to_verify_identity": [
 "Sorry you're having trouble verifying. Please make sure your document is valid and the photos are clear and well-lit, then try again.",
 "Verification can fail if the photo is blurry or the document is expired. Retry in good lighting, and contact us if it keeps failing.",
 "If identity verification isn't working, check that your details match your document. Our team can help if the problem continues."],
"verify_my_identity": [
 "You can verify your identity in the app by following the Verify identity steps, usually a photo of your ID and a selfie.",
 "To verify your identity, open the app and follow the prompts to upload your ID document and take a selfie.",
 "Identity verification is done in the app. Have a valid ID ready and follow the on-screen steps."],
"verify_source_of_funds": [
 "We may ask about the source of your funds to meet regulatory requirements. You can upload supporting documents in the app.",
 "Source of funds checks are required by regulation. Please provide the requested documents, such as a payslip or bank statement.",
 "To verify your source of funds, follow the request in the app and upload a document showing where the money came from."],
"verify_top_up": [
 "Some card top-ups need a quick verification. Follow the steps in the app to confirm the card belongs to you.",
 "We may ask you to verify a top-up card for security. The app will show you what's needed.",
 "Top-up verification helps keep your account safe. Complete the steps shown in the app to continue."],
"virtual_card_not_working": [
 "Sorry your virtual card isn't working. Check that it isn't frozen and that online payments are enabled in the card settings.",
 "Make sure you've entered the virtual card details correctly and the card isn't frozen. Some merchants may not accept virtual cards.",
 "Please check the card's status in the app. If it still doesn't work, try generating a new virtual card or contact us."],
"visa_or_mastercard": [
 "The card network depends on the card type and your country. You can see it on your card or in the card details in the app.",
 "Your card's network (Visa or Mastercard) is shown on the card and in the app. It may vary by country.",
 "Check the card details in the app to see which network your card uses."],
"why_verify_identity": [
 "We're required by law to verify our customers' identities to prevent fraud and keep accounts safe.",
 "Identity verification helps protect you and meets financial regulations. It only needs to be done once in most cases.",
 "Like all regulated financial providers, we must confirm who our customers are. This helps prevent fraud and misuse."],
"wrong_amount_of_cash_received": [
 "Sorry you received the wrong amount. Please contact us with the withdrawal details, and we'll investigate with the ATM operator.",
 "If the ATM gave you less cash than requested, report it to us with the time and location so we can raise a claim.",
 "Please let us know the transaction details. We'll look into it and work with the ATM operator to resolve it."],
"wrong_exchange_rate_for_cash_withdrawal": [
 "The rate for cash withdrawals can differ if the ATM converted the currency. Choosing to be charged in the local currency usually gets a better rate.",
 "Sorry about the rate. If the ATM offered conversion, its rate may have applied. Share the transaction details and we'll review it.",
 "Exchange rates for withdrawals depend on who did the conversion and when. Contact us if the rate looks wrong."],
}

if __name__ == "__main__":
    out = Path(__file__).parent
    assert len(R) == 77 and all(len(v) == 3 for v in R.values()), (len(R), [k for k, v in R.items() if len(v) != 3])
    with open(out / "banking77_replies.jsonl", "w", encoding="utf-8") as f:
        for intent, replies in R.items():
            for i, r in enumerate(replies):
                f.write(json.dumps({"intent": intent, "variant": i, "reply": r}) + "\n")
    (out / "banking77_labels.json").write_text(json.dumps(sorted(R), indent=0), encoding="utf-8")
    print(f"wrote {sum(len(v) for v in R.values())} replies for {len(R)} intents")
