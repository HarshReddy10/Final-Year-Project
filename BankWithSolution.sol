// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

contract Bank {
    uint private participantsLiquidity; // Balance of all participants
    uint private beforeOperation;
    uint private afterOperation;
    address private attacker;
    address private owner;
    mapping(address => uint256) private customerBalance;

    event ReentrancyAttackDetected(address attacker); // Event to log reentrancy

    constructor() payable {
        owner = msg.sender;
        customerBalance[msg.sender] += msg.value;
        participantsLiquidity = address(this).balance;
        beforeOperation = address(this).balance;
        afterOperation = 0;
    }

    modifier ownerOnly() {
        require(msg.sender == owner, "Caller is not the bank owner");
        _;
    }

    /** Store the attacker address, accessible only by the owner */
    function getAttackerAddress() external view ownerOnly returns (address) {
        return attacker;
    }

    /** Customer deposit function */
    function depositFunds() external payable returns (bool) {
        require(msg.value > 0, "Deposit value must be greater than zero");
        customerBalance[msg.sender] += msg.value;
        beforeOperation = getBankLiquidity();
        afterOperation = getBankLiquidity() - beforeOperation;
        participantsLiquidity += afterOperation;
        beforeOperation = getBankLiquidity();
        return true;
    }

    /** Customer withdraw function with improved reentrancy detection */
    function withdrawFunds(uint256 _value) public {
        require(_value <= customerBalance[msg.sender], "Insufficient account balance");

        // Check if the bank liquidity matches the total participants' liquidity
        if (getBankLiquidity() == getParticipantsLiquidity()) {
            // Update balances first to prevent reentrancy issues
            customerBalance[msg.sender] -= _value;
            participantsLiquidity -= _value;

            // Attempt to send Ether
            (bool sent, ) = msg.sender.call{value: _value}("");
            require(sent, "Failed to send Ether");
        } else {
            // Log attacker and emit an event
            attacker = msg.sender;
            emit ReentrancyAttackDetected(msg.sender);
            beforeOperation = getBankLiquidity();
        }
    }

    /** Transfer coins within the contract */
    function transfer(address to, uint256 amount) public {
        // Check if the bank liquidity matches the total participants' liquidity
        if (getBankLiquidity() == getParticipantsLiquidity()) {
            require(amount <= customerBalance[msg.sender], "Insufficient account balance");

            // Update balances
            customerBalance[to] += amount;
            customerBalance[msg.sender] -= amount;
        } else {
            // Log attacker and emit an event
            attacker = msg.sender;
            emit ReentrancyAttackDetected(msg.sender);
            beforeOperation = getBankLiquidity();
        }
    }

    /** Fetch bank liquidity */
    function getBankLiquidity() public view returns (uint256) {
        return address(this).balance;
    }

    /** Fetch participants liquidity */
    function getParticipantsLiquidity() public view returns (uint256) {
        return participantsLiquidity;
    }

    /** Fetch customer balance */
    function getCustomerBalance() public view returns (uint256) {
        return customerBalance[msg.sender];
    }
}
